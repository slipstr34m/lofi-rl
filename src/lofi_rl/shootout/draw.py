"""The job-time draw: which option each symbol stands for, the order of the rounds, and the hidden controls
(docs/SHOOTOUTS.md sections 2.3-2.5 and 7.6).

The repo is public, so nothing committed says which option is on which tile [CD-21]. The draw runs in the
`lofi-shootout` job from a 64-bit seed (`secrets.randbits(64)` unless a test passes one) and is recorded only in
`runs/shootout-<job>/summary/draw.json`; the batch builder later writes it into the hidden manifests.

What is drawn, per day:
1. each stage's symbol-to-option maps, within the stage's constraints: K1's twin pairs are always families
   (384 maps), plagal and float always take v2 or v3, the S1 tenors and altos are drawn within their register;
   a stage drawn from the job's leaders (K2, the S1 final) maps the ranked leaders onto its symbols at random;
2. the hidden controls (section 2.5): each anchor imitates a design round of its batch whose stage plays a sax
   (the same loop and candidates, tiles redrawn) with one drawn clip replaced by the sour copy rendered with
   that clip's candidate; on day 3 each repeat group shows one round of a `--prior` batch again, tiles
   reversed, never the first round served of a stage [CD-24], and only from the stages `repeats` names;
3. on day 3 the split of the pool (W1, W2, the S2 re-run, the anchor, the repeats) into the two mix batches of
   about equal planned length;
4. the order of the rounds in each batch [CD-22]: no two consecutive rounds on one loop, and a design round first.

Every draw uses SplitMix64 (`lofi_rl.prng`) keyed by the seed and a label (stage, batch or group), so one seed
gives the same draw on every machine and Python version, and adding a stage leaves the others' draws alone.

The draw also lists what the job must make: `renders` (render id -> loop text, brief, palette key, stage,
option) and `palettes` (palette key -> the band that palette plays: the generator's input, section 7.8).

Sound claims: none.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lofi_rl.prng import GOLDEN, MASK, below, splitmix64

from .plan import (BAND_SLOTS, SLOT_LIST, Batch, Plan, PlanError, Stage, apply_cuts, day_name, day_number,
                   substitute)
from .sour import check_sour, raise_accidentals, sax_voice

SCHEMA = "lofi-rl/shootout-draw/1"
BARS = 8
ORDER_TRIES = 4000              # random orders tried before the exhaustive search
SPLIT_TRIES = 500


class DrawError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ randomness
class Rng:
    """A SplitMix64 stream (`prng.stream`): output i is splitmix64(seed + i * GOLDEN)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    @classmethod
    def keyed(cls, seed: int, *labels: str) -> "Rng":
        """An independent stream for one purpose: the seed and the labels hashed together."""
        h = hashlib.sha256("|".join([f"{seed & MASK:016x}", *labels]).encode()).digest()
        return cls(int.from_bytes(h[:8], "big"))

    def next64(self) -> int:
        out = splitmix64(self.state)
        self.state = (self.state + GOLDEN) & MASK
        return out

    def below(self, k: int) -> int:
        return below(self.next64(), k)

    def shuffle(self, xs: list) -> list:
        for i in range(len(xs) - 1, 0, -1):                 # Fisher-Yates
            j = self.below(i + 1)
            xs[i], xs[j] = xs[j], xs[i]
        return xs

    def choice(self, xs: Sequence):
        return xs[self.below(len(xs))]

    def hex(self, n: int) -> str:
        return "".join("0123456789abcdef"[self.below(16)] for _ in range(n))


# ------------------------------------------------------------------------------------------------ inputs
def parse_leaders(text: str | None) -> dict[str, tuple[str, ...]]:
    """`keys=<id>+<id>+<id>/kit=<id>/bass=<id>/sax=<id>` (section 7.7) as {band slot: ids, leader first}."""
    out: dict[str, tuple[str, ...]] = {}
    for part in (text or "").split("/"):
        if not part:
            continue
        slot, eq, ids = part.partition("=")
        if not eq or slot not in SLOT_LIST or not ids:
            raise DrawError(f"--leaders: {part!r} is not <slot>=<id>[+<id>...] with a slot of {list(BAND_SLOTS)}")
        if slot in out:
            raise DrawError(f"--leaders: {slot} is given twice")
        out[slot] = tuple(ids.split("+"))
        if len(set(out[slot])) != len(out[slot]) or "" in out[slot]:
            raise DrawError(f"--leaders: {slot} lists an id twice or an empty id")
    return out


@dataclass(frozen=True)
class Prior:
    """A previous job's batches (its summary/batches.json) and draw, for day-3 repeats."""

    run_dir: Path
    batches: Mapping[str, Any]
    draw: Mapping[str, Any]

    @property
    def run(self) -> str:
        return self.run_dir.name


def load_prior(batches_json: Path | str) -> Prior:
    """`<run>/summary/batches.json` and the draw.json next to it."""
    p = Path(batches_json)
    try:
        batches = json.loads(p.read_text(encoding="utf-8"))
        draw = json.loads((p.parent / "draw.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise DrawError(f"--prior {p}: {err}") from err
    if batches.get("schema") != "lofi-rl/shootout-batches/1" or draw.get("schema") != SCHEMA:
        raise DrawError(f"--prior {p}: not a shootout batches.json with its draw.json")
    return Prior(p.parent.parent.resolve(), batches, draw)


# ------------------------------------------------------------------------------------------------ the draw
@dataclass
class _Ctx:
    plan: Plan
    day: str
    seed: int
    band: dict[str, str | None]
    renders: dict[str, dict]
    palettes: dict[str, dict]
    sour: dict[str, dict]
    texts: dict[str, str]

    def text(self, loop_id: str) -> str:
        if loop_id not in self.texts:
            self.texts[loop_id] = self.plan.loops[loop_id].text
        return self.texts[loop_id]


def _brief(b) -> dict:
    return {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic}


def round_seconds(clips: int, bpm: float) -> float:
    """The plan's listening estimate (taste_pipeline.md section 8): 0.75 of auto-play plus 8 s to mark and save."""
    return 0.75 * clips * BARS * 240.0 / bpm + 8.0


def est_minutes(rounds: Sequence[Mapping]) -> float:
    return round(sum(round_seconds(len(r["clips"]), r["bpm"]) for r in rounds) / 60.0, 1)


def _day_band(plan: Plan, day: str, leaders: Mapping[str, tuple[str, ...]], stages: Sequence[Stage]
              ) -> dict[str, str | None]:
    """The day's fixed band, with the leaders the job named (leader first) in their slots."""
    spec = plan.days.get(day_number(day))
    if spec is None:
        raise DrawError(f"the plan has no band for {day}")
    extra = sorted(set(leaders) - set(spec.leaders))
    if extra:
        raise DrawError(f"--leaders names {extra}, which {day} does not take (it takes {sorted(spec.leaders)})")
    counts: dict[str, set[int]] = {}
    for st in stages:
        for g in st.groups:
            if g.leaders:
                counts.setdefault(g.leaders, set()).update(st.slots_by_count)
    band = spec.band_slots()
    for slot, default in spec.leaders.items():
        ids = leaders.get(slot)
        if ids is None:
            if default is None:
                raise DrawError(f"{day} needs --leaders {slot}=<id>")
            ids = (default,)
        allowed = counts.get(slot, {1})
        if len(ids) not in allowed:
            raise DrawError(f"--leaders {slot}: {len(ids)} ids given, {day} takes {sorted(allowed)}")
        for i in ids:
            if i not in plan.band_options[slot]:
                raise DrawError(f"--leaders {slot}: {i!r} is not one of the {SLOT_LIST[slot]} candidates")
        band[slot] = ids[0]
    return band


def _map_group(plan: Plan, stage: Stage, gi: int, leaders: Mapping[str, tuple[str, ...]], rng: Rng
               ) -> dict[str, str]:
    g = stage.groups[gi]
    if g.pairs:                                                  # K1: each pair is one family [CD-8]
        fams: dict[Any, list[str]] = {}
        for o in g.options:
            fams.setdefault(getattr(plan.candidates[o], g.by), []).append(o)
        members = rng.shuffle(list(fams.values()))
        out = {}
        for (a, b), pair in zip(g.pairs, members):
            x, y = rng.shuffle(list(pair))
            out[a], out[b] = x, y
        return out
    opts = list(leaders[g.leaders]) if g.leaders else list(g.options)
    symbols = list(g.symbols[:len(opts)])
    out: dict[str, str] = {}
    free = list(symbols)
    for o in sorted(g.pin):                                      # pinned options first, within their symbols
        allowed = [s for s in g.pin[o] if s in free]
        if not allowed:
            raise DrawError(f"stage {stage.id}: no symbol left for the pin of {o}")
        s = rng.choice(allowed)
        out[s] = o
        free.remove(s)
    rest = rng.shuffle([o for o in opts if o not in g.pin])
    for s, o in zip(sorted(free, key=symbols.index), rest):
        out[s] = o
    return {s: out[s] for s in symbols}


def _text(ctx: _Ctx, stage: Stage, loop_id: str, option: str, sour: bool) -> tuple[str, str]:
    text = ctx.text(loop_id)
    variant = "base"
    if stage.sub:                               # K2's `keys: ep spread`, or the vamp's own line (section 5)
        text = substitute(text, **{head: ctx.plan.vamps[option].line if value == "option" else value
                                   for head, value in stage.sub.items()})
        variant = "vamp" if "vamp" in stage.sub else stage.sub["keys"].split()[-1]
    if sour:
        text, variant = raise_accidentals(text), "sour"
    return text, variant


def _clip(ctx: _Ctx, stage: Stage, loop_id: str, option: str, band: Mapping[str, str | None], *,
          sour: bool = False, kit: str | None = None) -> dict:
    """Register the render (and its palette) one clip needs; return the clip as the round lists it."""
    plan, lp = ctx.plan, ctx.plan.loops[loop_id]
    pal_band = dict(band)
    pal_band.update(stage.band)
    if kit:
        pal_band["kit"] = kit
    if stage.fills:
        pal_band[stage.fills] = option
    key = "_".join([stage.id.lower()] + ([kit] if kit else []) + ([option] if stage.fills else []))
    spec = {"stage": stage.id, "day": ctx.day, "fills": stage.fills, "option": option if stage.fills else None,
            "band": pal_band, "sax_muted": stage.sax_muted}
    if ctx.palettes.setdefault(key, spec) != spec:
        raise DrawError(f"palette {key} is asked for with two different bands")
    register = plan.candidates[pal_band["sax"]].register if pal_band.get("sax") else None
    text, variant = _text(ctx, stage, loop_id, option, sour)
    rid = f"{stage.id}.{lp.name}.{option}" + (".sour" if sour else "")
    entry = {"loop": lp.name, "brief": _brief(lp.brief), "text": text, "variant": variant, "palette": key,
             "stage": stage.id, "option": option, "what": "sour copy" if sour else stage.what, "bus": stage.bus,
             "register": register}
    if ctx.renders.setdefault(rid, entry) != entry:
        raise DrawError(f"render {rid} is asked for twice with different inputs")
    if sour:
        chk = check_sour(text, lp.brief, sax_voice(register, plan.registers[register]))
        rec = {"loop": lp.name, "stage": stage.id, "option": option, "register": register, **chk.to_json()}
        ctx.sour[rid] = rec
        if not chk.ok:
            raise DrawError(f"the sour copy of {lp.name} for {option} fails its gate: {rec}")
    clip = {"option": option, "render": rid, "what": entry["what"]}
    if kit:
        clip["kit"] = kit
    return clip


def _stage_rounds(ctx: _Ctx, stage: Stage, maps: list[dict[str, str]], band: Mapping[str, str | None],
                  count: int | None, other_maps: Mapping[str, list[dict[str, str]]]) -> list[dict]:
    rounds = []
    for slot in stage.slots_for(count):
        kit = None
        if slot.kit:                                # R2: the kit the day's kit stage's symbol stands for
            ref = next(sid for sid in other_maps if ctx.plan.stages[sid].fills == "kit")
            kit = next(m[slot.kit] for m in other_maps[ref] if slot.kit in m)
        clips = [_clip(ctx, stage, slot.loop, maps[stage.group_of(slot.id, sym)][sym], band, kit=kit)
                 for sym in slot.tiles]
        lp = ctx.plan.loops[slot.loop]
        rounds.append({"kind": "none", "stage": stage.id, "slot": slot.id, "loop": lp.name, "bpm": lp.brief.bpm,
                       "category": stage.category, "clips": clips})
    return rounds


def _anchor(ctx: _Ctx, rounds: list[dict], band_of: Mapping[str, Mapping], rng: Rng) -> dict:
    """An extra round imitating a drawn design round with a sax: same loop and candidates, tiles redrawn, one
    drawn clip replaced by the sour copy rendered with that clip's candidate (section 2.5, CD-23)."""
    pool = [r for r in rounds if r["kind"] == "none" and ctx.plan.stages[r["stage"]].anchor]
    if not pool:
        raise DrawError("an anchor needs a design round whose stage plays a sax")
    src = rng.choice(pool)
    stage = ctx.plan.stages[src["stage"]]
    clips = rng.shuffle([dict(c) for c in src["clips"]])
    pos = rng.below(len(clips))
    victim = clips[pos]
    clips[pos] = _clip(ctx, stage, ctx.plan.loop_named(src["loop"]).id, victim["option"], band_of[stage.id],
                       sour=True, kit=victim.get("kit"))
    return {**{k: v for k, v in src.items() if k != "clips"}, "kind": "anchor", "imitates": src["slot"],
            "anchor_pos": pos + 1, "clips": clips}


def _ok_order(rounds: Sequence[Mapping]) -> bool:
    return bool(rounds) and rounds[0]["kind"] == "none" and all(
        a["loop"] != b["loop"] for a, b in zip(rounds, rounds[1:]))


def order_rounds(rounds: list[dict], rng: Rng) -> list[dict]:
    """A random order with a design round first and no two consecutive rounds on one loop [CD-22]."""
    for _ in range(ORDER_TRIES):
        cand = rng.shuffle(list(rounds))
        if _ok_order(cand):
            return cand
    # exhaustive search in a random branch order: the doc's planner found orders in every draw it tried
    def dfs(prefix: list[dict], left: list[dict]):
        if not left:
            return prefix
        for i in rng.shuffle(list(range(len(left)))):
            r = left[i]
            if (not prefix and r["kind"] != "none") or (prefix and prefix[-1]["loop"] == r["loop"]):
                continue
            got = dfs(prefix + [r], left[:i] + left[i + 1:])
            if got:
                return got
        return None
    got = dfs([], list(rounds))
    if got is None:
        raise DrawError("no order puts a design round first with no two consecutive rounds on one loop")
    return got


def _repeats(ctx: _Ctx, priors: Sequence[Prior], stages: Sequence[str], rng_of) -> list[dict]:
    """One repeat per group (section 2.5), from the eligible stages' rounds of the prior batches."""
    rounds = []                                     # (prior, batch name, round) in serving order
    for pr in priors:
        for bname in sorted(pr.batches["batches"]):
            for r in pr.batches["batches"][bname]["rounds"]:
                rounds.append((pr, bname, r))
    first: dict[str, tuple] = {}
    for pr, bname, r in sorted(rounds, key=lambda x: (x[1], x[2]["index"])):
        if r["kind"] == "none":
            first.setdefault(r["stage"], (pr.run, bname, r["id"]))
    out = []
    for group, members in ctx.plan.repeat_groups.items():
        eligible = [s for s in members if s in stages]
        pool = [(pr, b, r) for pr, b, r in rounds if r["kind"] == "none" and r["stage"] in eligible
                and first.get(r["stage"]) != (pr.run, b, r["id"])]
        if not pool:
            continue
        pr, bname, src = rng_of(group).choice(pool)
        clips = []
        for c in reversed(src["clips"]):           # tiles reversed (D-013)
            clips.append(_prior_clip(ctx, pr, c))
        lp = ctx.plan.loop_named(src["loop"])
        out.append({"kind": "repeat", "stage": src["stage"], "slot": src["slot"], "loop": src["loop"],
                    "bpm": lp.brief.bpm, "category": src["category"], "group": group, "of": src["id"],
                    "of_batch": bname, "of_run": pr.run, "clips": clips})
    return out


def _prior_clip(ctx: _Ctx, pr: Prior, c: Mapping) -> dict:
    """A repeat's clip: the first showing's render, or a fresh render of it when scratch has purged it."""
    run_dir = Path(c.get("run_dir") or pr.run_dir)
    rid = c["render"]
    expect = {"clip": c["clip"], "pcm_sha256": c["pcm_sha256"], "flac_sha256": c["flac_sha256"]}
    clip = {"option": c["option"], "what": c.get("what"), "expect": expect}
    if c.get("kit"):
        clip["kit"] = c["kit"]
    if (run_dir / "renders" / rid / "render.flac").is_file():
        return {**clip, "render": rid, "run_dir": str(run_dir)}
    src_draw = pr.draw if run_dir == pr.run_dir else _read_draw(run_dir)
    entry = src_draw.get("renders", {}).get(rid)
    pal = run_dir / "summary" / "palettes" / f"{entry['palette']}.yaml" if entry else None
    if entry is None or not pal.is_file():
        raise DrawError(f"render {rid} of {run_dir} was purged and cannot be made again (no draw entry or palette)")
    new = f"prior.{run_dir.name}.{rid}"
    ctx.renders[new] = {**entry, "palette_path": str(pal), "expect_pcm_sha256": c["pcm_sha256"]}
    return {**clip, "render": new}


def _read_draw(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "summary" / "draw.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise DrawError(f"{run_dir}: no readable summary/draw.json ({err})") from err


def _split(rounds: list[dict], names: Sequence[str], rng: Rng) -> dict[str, list[dict]]:
    """The pool split into the batches by planned length (longest first onto the shorter batch, ties drawn),
    redrawn until every batch can open with a design round and be ordered."""
    for _ in range(SPLIT_TRIES):
        pool = rng.shuffle(list(rounds))
        pool.sort(key=lambda r: -round_seconds(len(r["clips"]), r["bpm"]))
        out = {n: [] for n in names}
        tot = {n: 0.0 for n in names}
        for r in pool:
            low = min(tot.values())
            n = rng.choice([x for x in names if tot[x] == low])
            out[n].append(r)
            tot[n] += round_seconds(len(r["clips"]), r["bpm"])
        if all(any(r["kind"] == "none" for r in v) for v in out.values()):
            return out
    raise DrawError(f"could not split {len(rounds)} rounds into {list(names)} with a design round in each")


def draw(plan: Plan, day: str, *, seed: int, leaders: Mapping[str, tuple[str, ...]] | None = None,
         priors: Sequence[Prior] = (), repeats: Sequence[str] = (), rerun_s2: str | None = None,
         cuts: Sequence[str] = ()) -> dict:
    """The day's draw as draw.json holds it. Raises DrawError (or PlanError) before anything is written."""
    leaders = dict(leaders or {})
    n = day_number(day)
    day = day_name(n)
    by_id = {c.id: c for c in plan.cuts}
    for c in cuts:
        if c in by_id and by_id[c].day != n:
            raise DrawError(f"cut {c} belongs to {day_name(by_id[c].day)}, not {day}")
    bad = [s for s in repeats if s not in plan.stages]
    if bad:
        raise DrawError(f"--repeats names {bad}, which are not stages of the plan")
    plan = apply_cuts(plan, list(cuts))
    if not plan.batches_of(n):
        raise DrawError(f"the plan has no batches for {day}")
    if repeats and n != plan.repeat_day:
        raise DrawError(f"repeats come only on {day_name(plan.repeat_day)}")
    if repeats and not priors:
        raise DrawError("repeats need the --prior batches they repeat")
    if rerun_s2 and n != plan.rerun_day:
        raise DrawError("the S2 re-run comes only on " + (day_name(plan.rerun_day) if plan.rerun_day else "no day"))
    stages = list(plan.stages_of(n))
    band = _day_band(plan, day, leaders, stages)
    if rerun_s2 and rerun_s2 not in plan.band_options["sax"]:
        raise DrawError(f"--rerun-s2: {rerun_s2!r} is not one of the saxes")
    ctx = _Ctx(plan, day, seed, band, {}, {}, {}, {})
    maps: dict[str, list[dict[str, str]]] = {}
    by_stage: dict[str, list[dict]] = {}
    band_of: dict[str, dict] = {}
    run_stages = [(st, st.id, band) for st in stages]
    if rerun_s2:
        rerun_band = {**band, "sax": rerun_s2}
        run_stages += [(replace(plan.stages[s], day=n), s, rerun_band) for s in plan.rerun_stages]
    for st, sid, b in run_stages:
        if sid in maps:
            raise DrawError(f"stage {sid} runs twice on {day}")
        rng = Rng.keyed(seed, "map", day, sid)
        maps[sid] = [_map_group(plan, st, gi, leaders, rng) for gi in range(len(st.groups))]
        count = len(leaders[st.groups[0].leaders]) if st.groups[0].leaders else None
        band_of[sid] = b
        by_stage[sid] = _stage_rounds(ctx, st, maps[sid], b, count, maps)
    rerun_ids = [s for _, s, _ in run_stages if rerun_s2 and s in plan.rerun_stages]
    batches: list[dict] = []
    for b in plan.batches_of(n):
        if b.split is None:                     # one batch: its stages' rounds and its anchors
            (name,) = b.names
            rounds = [r for s in b.stages for r in by_stage[s]]
            rounds += _anchors(ctx, b, name, rounds, band_of, seed)
            batches.append(_batch_doc(b, name, rounds, Rng.keyed(seed, "order", name)))
            continue
        # a split pool (day 3): its stages, the S2 re-run, the anchor and the repeats, divided by planned length
        name = b.split
        stage_ids = list(dict.fromkeys(b.stages)) + rerun_ids
        pool = [r for s in stage_ids for r in by_stage[s]]
        pool += _anchors(ctx, b, name, pool, band_of, seed)
        pool += _repeats(ctx, priors, repeats, lambda g: Rng.keyed(seed, "repeat", g))
        rng = Rng.keyed(seed, "split", name)
        for _ in range(SPLIT_TRIES):
            parts = _split(pool, list(b.names), rng)
            try:
                docs = [_batch_doc(b, x, parts[x], Rng.keyed(seed, "order", x)) for x in b.names]
                break
            except DrawError:
                continue
        else:
            raise DrawError(f"no split of {name} can be ordered")
        batches += docs
    batches.sort(key=lambda d: (d["date"], d["name"]))
    return {
        "schema": SCHEMA, "v": 1, "stage": day, "seed": f"{seed & MASK:016x}",
        "plan": {"path": str(plan.path), "sha256": plan.sha256, "cuts": list(plan.applied_cuts)},
        "leaders": {k: list(v) for k, v in sorted(leaders.items())}, "band": band,
        "repeats": {"stages": list(repeats), "prior": [str(p.run_dir) for p in priors]},
        "rerun_s2": rerun_s2,
        "maps": {sid: maps[sid] for sid in maps},
        "batches": batches,
        "renders": dict(sorted(ctx.renders.items())),
        "palettes": dict(sorted(ctx.palettes.items())),
        "sour": dict(sorted(ctx.sour.items())),
    }


def _anchors(ctx: _Ctx, b: Batch, name: str, rounds: list[dict], band_of: Mapping[str, Mapping], seed: int
             ) -> list[dict]:
    """The batch's (or split pool's) anchor rounds (section 2.5: one per day), each drawn from its own stream."""
    return [_anchor(ctx, rounds, band_of, Rng.keyed(seed, "anchor", name if k == 0 else f"{name}#{k + 1}"))
            for k in range(b.anchors)]


def _batch_doc(b: Batch, name: str, rounds: list[dict], rng: Rng) -> dict:
    ordered = order_rounds(rounds, rng)
    return {"name": name, "date": b.date, "day": day_name(b.day), "category": b.category, "note": b.note,
            "est_min": est_minutes(ordered), "rounds": ordered}


def write_draw(doc: Mapping, path: Path | str) -> Path:
    """draw.json, written atomically (sorted keys, so one seed gives the same bytes)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


__all__ = ["DrawError", "Prior", "Rng", "SCHEMA", "draw", "est_minutes", "load_prior", "order_rounds",
           "parse_leaders", "round_seconds", "write_draw", "PlanError"]
