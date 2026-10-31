"""The shootout plan in symbols: shootouts/plan.yaml (docs/SHOOTOUTS.md sections 4, 6 and 7.6).

The plan is public (section 2.3), so it holds the candidate lists, each stage's design in symbols with its
constraints, the loop of each round slot and each batch's contents, and never places an option on a tile. Which
option a symbol stands for is drawn at job time (`lofi_rl.shootout.draw`).

Loading is strict where the batch builder reads the plan: unknown keys, a tile that is not a declared symbol, an
option id on a tile, a slot on an unknown loop, a group whose symbols do not match its options, a batch naming an
unknown stage, are each a `PlanError`. Candidate entries are free mappings beyond the fields the builder reads
(`family`, `register`, `mood`, `line`): the rest is for the palette generator (section 7.8) and the reader.

    from lofi_rl.shootout.plan import load_plan, apply_cuts
    plan = apply_cuts(load_plan(), ["latin3"])
    plan.stages["K1"].slots[0].tiles        # ("P1a", "P1b", "P2a", "P2b")
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from lofi_rl.brief import SCENE, Brief
from lofi_rl.palette import ROOT
from lofi_rl.ui.manifest import BATCH_RE

PLAN = ROOT / "shootouts" / "plan.yaml"
DAYS = ("day1", "day2", "day3")
# The band: every slot a clip's palette fills (section 4 "fixed in every round", CD-7), and the candidate list
# each slot draws from.
SLOT_LIST: Mapping[str, str] = MappingProxyType({
    "keys": "keys", "kit": "kits", "bass": "basses", "sax": "saxes", "swing": "swing", "humanize": "humanize",
    "sax_variant": "sax_variants", "timing": "timing", "bed": "excerpts", "rain": "rain"})
BAND_SLOTS = tuple(SLOT_LIST)
BUSES = ("keys", "drums", "bass", "sax", "bed", "keys+bass")
SUBS = ("spread", "vamp")               # K2's `keys: ep spread`; the vamp rounds' `vamp:` line (section 5)
MAX_TILES = 4                           # the page's limit (RATING.md: a shootout round has 2-4 clips)

_ID = re.compile(r"[a-z][a-z0-9_]*")
_STAGE = re.compile(r"[A-Z][A-Za-z0-9]*")
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_LOOP = re.compile(r"L[0-9]+")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_TOP = ("v", "scene", "loops_dir", "loops", "candidates", "band", "stages", "batches")
_TOP_OPT = ("alto_above", "registers", "repeats", "rerun_s2", "cuts")
_STAGE_KEYS = ("day", "category", "fills", "bus", "draw")
_STAGE_OPT = ("what", "slots", "slots_by_count", "band", "default", "sub", "sax_muted", "anchor", "repeat_group")


class PlanError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ types
@dataclass(frozen=True)
class Loop:
    id: str                     # L1 .. L9
    name: str                   # the file stem, e.g. l1_melancholy_low; the hidden manifests' brief_id
    path: Path
    brief: Brief

    @property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class Group:
    """One draw group: its symbols stand for its options, one each, drawn at job time."""

    symbols: tuple[str, ...]
    options: str                                # a candidate list, or "leaders.<band slot>"
    where: Mapping[str, Any]                    # a filter on the list's entries
    pin: Mapping[str, tuple[str, ...]]          # option -> the symbols it may take
    pairs: tuple[tuple[str, str], ...] = ()     # twin draws: each pair takes the two options of one `by` value
    by: str | None = None
    slots: tuple[str, ...] = ()                 # the slots this group's symbols apply to (empty: every slot)

    @property
    def leaders(self) -> str | None:
        """The band slot whose job-time leaders this group draws from, or None for a candidate list."""
        return self.options.split(".", 1)[1] if self.options.startswith("leaders.") else None

    def covers(self, slot: str) -> bool:
        return not self.slots or slot in self.slots


@dataclass(frozen=True)
class Slot:
    id: str                     # e.g. K1.3
    loop: str                   # L1 .. L9
    tiles: tuple[str, ...]      # symbols, tile 1 first
    kit: str | None = None      # "R1.k1": this round plays the kit another stage's symbol stands for (R2)


@dataclass(frozen=True)
class Stage:
    id: str
    day: str
    category: str
    fills: str | None           # the band slot the option takes; None: the option is a loop substitution
    bus: str
    what: str | None
    groups: tuple[Group, ...]
    slots: tuple[Slot, ...]
    slots_by_count: Mapping[int, tuple[Slot, ...]]
    band: Mapping[str, str | None]
    default: str | None
    sub: str | None
    sax_muted: bool
    anchor: bool                # an anchor may imitate this stage's rounds (it plays a sax)
    repeat_group: str | None

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(s for g in self.groups for s in g.symbols)

    def slots_for(self, count: int | None = None) -> tuple[Slot, ...]:
        """The round slots; for a stage drawn from the job's leaders, those for `count` options."""
        if not self.slots_by_count:
            return self.slots
        if count not in self.slots_by_count:
            raise PlanError(f"stage {self.id} has slots for {sorted(self.slots_by_count)} options, not {count}")
        return self.slots_by_count[count]

    def group_of(self, slot: str, symbol: str) -> int:
        """Index of the group that resolves `symbol` in `slot` (validated to be exactly one at load)."""
        hits = [i for i, g in enumerate(self.groups) if symbol in g.symbols and g.covers(slot)]
        if len(hits) != 1:
            raise PlanError(f"stage {self.id} slot {slot}: symbol {symbol!r} resolves to {len(hits)} groups")
        return hits[0]


@dataclass(frozen=True)
class Band:
    fixed: Mapping[str, str | None]
    leaders: Mapping[str, str | None]           # slot -> its default, or None when the job must name it


@dataclass(frozen=True)
class Batch:
    name: str
    date: str
    day: str
    category: str
    note: str
    stages: tuple[str, ...]
    anchor: bool
    split: str | None


@dataclass(frozen=True)
class CutEdit:
    stage: str
    keep: tuple[str, ...] | None
    tiles: Mapping[str, tuple[str, ...]]
    pin: Mapping[str, tuple[str, ...]]
    drop: bool


@dataclass(frozen=True)
class Cut:
    name: str
    day: str
    edits: tuple[CutEdit, ...]


@dataclass(frozen=True)
class Plan:
    path: Path
    sha256: str
    scene: str
    loops: Mapping[str, Loop]
    alto_above: frozenset[str]
    registers: Mapping[str, Mapping[str, tuple[int, int]]]
    candidates: Mapping[str, Mapping[str, Mapping[str, Any]]]
    band: Mapping[str, Band]
    stages: Mapping[str, Stage]
    batches: tuple[Batch, ...]
    repeat_day: str | None
    repeat_groups: Mapping[str, tuple[str, ...]]
    rerun_day: str | None
    rerun_stages: tuple[str, ...]
    cuts: Mapping[str, Cut]
    applied_cuts: tuple[str, ...] = ()

    def options(self, lst: str, where: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        """Option ids of a candidate list, in file order, filtered by `where`."""
        entries = self.candidates[lst]
        return tuple(o for o, e in entries.items() if all(e.get(k) == v for k, v in (where or {}).items()))

    def list_of(self, option: str) -> str:
        for lst, entries in self.candidates.items():
            if option in entries:
                return lst
        raise PlanError(f"{option!r} is not a candidate of the plan")

    def loop_named(self, name: str) -> Loop:
        for lp in self.loops.values():
            if lp.name == name:
                return lp
        raise PlanError(f"no loop is named {name!r}")

    def stages_of(self, day: str) -> tuple[Stage, ...]:
        return tuple(s for s in self.stages.values() if s.day == day)

    def batches_of(self, day: str) -> tuple[Batch, ...]:
        return tuple(b for b in self.batches if b.day == day)


# ------------------------------------------------------------------------------------------------ loading
def load_plan(path: Path | str = PLAN) -> Plan:
    """Load and validate a plan file. Raises PlanError naming the first problem."""
    path = Path(path)
    data = path.read_bytes()
    try:
        doc = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as err:
        raise PlanError(f"{path}: not a UTF-8 YAML document ({err})") from err
    return parse_plan(doc, path, hashlib.sha256(data).hexdigest())


def _fail(where: str, what: str) -> PlanError:
    return PlanError(f"{where}: {what}")


def _map(node: Any, where: str, required: tuple[str, ...] = (), optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(node, dict):
        raise _fail(where, "expected a mapping")
    missing = [k for k in required if k not in node]
    if missing:
        raise _fail(where, f"missing {missing}")
    extra = sorted(str(k) for k in node if k not in required + optional)
    if extra and (required or optional):
        raise _fail(where, f"unknown key(s) {extra}")
    return node


def _str(node: Any, where: str, pattern: re.Pattern | None = None) -> str:
    if not isinstance(node, str) or not node.strip():
        raise _fail(where, f"expected text, got {node!r}")
    if pattern is not None and not pattern.fullmatch(node):
        raise _fail(where, f"{node!r} does not match {pattern.pattern}")
    return node


def _list(node: Any, where: str) -> list:
    if not isinstance(node, list) or not node:
        raise _fail(where, "expected a non-empty list")
    return node


def parse_plan(doc: Any, path: Path, sha256: str) -> Plan:
    top = _map(doc, "plan", _TOP, _TOP_OPT)
    if top["v"] != 1:
        raise _fail("v", "must be 1")
    if top["scene"] != SCENE:
        raise _fail("scene", f"must be {SCENE!r} (the only scene, TASTE.md)")
    loops = _loops(top, path)
    candidates = _candidates(top["candidates"])
    owner: dict[str, str] = {}
    for lst, entries in candidates.items():
        for o in entries:
            if o in owner:
                raise _fail(f"candidates.{lst}.{o}", f"appears twice (also in {owner[o]})")
            owner[o] = lst
    registers = {}
    for reg, raw in (top.get("registers") or {}).items():
        m = _map(raw, f"registers.{reg}", ("play", "comfort", "home"))
        registers[reg] = MappingProxyType({k: _pair(m[k], f"registers.{reg}.{k}") for k in ("play", "comfort", "home")})
    alto_above = frozenset(top.get("alto_above") or ())
    for lp in alto_above:
        if lp not in loops:
            raise _fail("alto_above", f"{lp!r} is not a loop")
    ctx = _Ctx(loops, candidates, owner)
    band = {}
    for day, raw in _map(top["band"], "band").items():
        if day not in DAYS:
            raise _fail(f"band.{day}", f"not one of {list(DAYS)}")
        m = _map(raw, f"band.{day}", (), ("fixed", "leaders"))
        band[day] = Band(MappingProxyType(ctx.band(m.get("fixed") or {}, f"band.{day}.fixed")),
                         MappingProxyType(ctx.band(m.get("leaders") or {}, f"band.{day}.leaders")))
    stages = {}
    for sid, raw in _map(top["stages"], "stages").items():
        _str(sid, "stages key", _STAGE)
        stages[sid] = _stage(sid, raw, ctx)
    for st in stages.values():
        for slots in [st.slots, *st.slots_by_count.values()]:
            for slot in slots:
                if slot.kit:
                    ref, _, sym = slot.kit.partition(".")
                    if ref not in stages or stages[ref].fills != "kit" or sym not in stages[ref].symbols:
                        raise _fail(f"stages.{st.id}.{slot.id}.kit", f"{slot.kit!r} is not a kit symbol of a stage")
    batches = []
    for i, raw in enumerate(_list(top["batches"], "batches")):
        batches.append(_batch(raw, f"batches[{i}]", stages))
    names = [b.name for b in batches]
    if len(set(names)) != len(names):
        raise _fail("batches", "a batch name appears twice")
    rep = _map(top.get("repeats") or {"day": None, "groups": {}}, "repeats", ("day", "groups"))
    groups = {}
    for g, sids in (rep["groups"] or {}).items():
        groups[g] = tuple(_stage_ref(s, f"repeats.groups.{g}", stages) for s in _list(sids, f"repeats.groups.{g}"))
    rr = _map(top.get("rerun_s2") or {"day": None, "stages": []}, "rerun_s2", ("day", "stages"))
    rerun = tuple(_stage_ref(s, "rerun_s2.stages", stages) for s in rr["stages"] or ())
    cuts = {}
    for name, raw in (top.get("cuts") or {}).items():
        cuts[name] = _cut(name, raw, stages)
    return Plan(path=path, sha256=sha256, scene=top["scene"], loops=MappingProxyType(loops), alto_above=alto_above,
                registers=MappingProxyType(registers), candidates=MappingProxyType(candidates),
                band=MappingProxyType(band), stages=MappingProxyType(stages), batches=tuple(batches),
                repeat_day=rep["day"], repeat_groups=MappingProxyType(groups), rerun_day=rr["day"],
                rerun_stages=rerun, cuts=MappingProxyType(cuts))


def _pair(node: Any, where: str) -> tuple[int, int]:
    if not (isinstance(node, list) and len(node) == 2 and all(isinstance(x, int) for x in node) and node[0] <= node[1]):
        raise _fail(where, "expected [low, high] whole numbers")
    return node[0], node[1]


def _loops(top: dict, path: Path) -> dict[str, Loop]:
    base = Path(_str(top["loops_dir"], "loops_dir"))
    base = base if base.is_absolute() else path.parent / base
    briefs_path = base / "briefs.yaml"
    try:
        briefs = yaml.safe_load(briefs_path.read_text(encoding="utf-8"))["loops"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as err:
        raise _fail("loops_dir", f"cannot read {briefs_path} ({err})") from err
    out = {}
    for lid, name in _map(top["loops"], "loops").items():
        _str(lid, "loops key", _LOOP)
        _str(name, f"loops.{lid}", _ID)
        file = base / f"{name}.ls"
        if not file.is_file():
            raise _fail(f"loops.{lid}", f"{file} is missing")
        b = briefs.get(name) if isinstance(briefs, dict) else None
        if not isinstance(b, dict):
            raise _fail(f"loops.{lid}", f"{name} has no brief in {briefs_path}")
        try:
            brief = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"])
        except (KeyError, ValueError) as err:
            raise _fail(f"loops.{lid}", f"bad brief ({err})") from err
        out[lid] = Loop(lid, name, file, brief)
    if len({lp.name for lp in out.values()}) != len(out):
        raise _fail("loops", "a loop is listed twice")
    return out


def _candidates(node: Any) -> dict[str, Mapping[str, Mapping[str, Any]]]:
    out = {}
    for lst, entries in _map(node, "candidates").items():
        _str(lst, "candidates key", _ID)
        rows = {}
        for o, e in _map(entries, f"candidates.{lst}").items():
            _str(o, f"candidates.{lst} key", _ID)
            if e is not None and not isinstance(e, dict):
                raise _fail(f"candidates.{lst}.{o}", "expected a mapping of its fields")
            rows[o] = MappingProxyType(dict(e or {}))
        if not rows:
            raise _fail(f"candidates.{lst}", "is empty")
        out[lst] = MappingProxyType(rows)
    return out


class _Ctx:
    def __init__(self, loops, candidates, owner):
        self.loops, self.candidates, self.owner = loops, candidates, owner

    def band(self, node: Any, where: str) -> dict[str, str | None]:
        out = {}
        for slot, opt in _map(node, where).items():
            if slot not in SLOT_LIST:
                raise _fail(f"{where}.{slot}", f"not a band slot {list(BAND_SLOTS)}")
            if opt is not None and opt not in self.candidates.get(SLOT_LIST[slot], {}):
                raise _fail(f"{where}.{slot}", f"{opt!r} is not one of the {SLOT_LIST[slot]} candidates")
            out[slot] = opt
        return out

    def options(self, spec: str, where_filter: Mapping, where: str) -> tuple[str, ...] | None:
        """The group's options, or None when they come from the job's leaders."""
        if spec.startswith("leaders."):
            slot = spec.split(".", 1)[1]
            if slot not in SLOT_LIST:
                raise _fail(where, f"{spec!r} names no band slot")
            return None
        if spec not in self.candidates:
            raise _fail(where, f"{spec!r} is not a candidate list")
        opts = tuple(o for o, e in self.candidates[spec].items()
                     if all(e.get(k) == v for k, v in where_filter.items()))
        if not opts:
            raise _fail(where, f"no {spec} candidate matches {dict(where_filter)}")
        return opts


def _stage_ref(sid: Any, where: str, stages: Mapping[str, Stage]) -> str:
    if sid not in stages:
        raise _fail(where, f"{sid!r} is not a stage")
    return sid


def _group(raw: Any, where: str, ctx: _Ctx, slot_ids: set[str]) -> Group:
    m = _map(raw, where, ("options",), ("symbols", "pairs", "by", "where", "pin", "slots"))
    if ("symbols" in m) == ("pairs" in m):
        raise _fail(where, "give either symbols or pairs")
    filt = MappingProxyType(dict(_map(m.get("where") or {}, f"{where}.where")))
    opts = ctx.options(_str(m["options"], f"{where}.options"), filt, f"{where}.options")
    pairs: tuple[tuple[str, str], ...] = ()
    if "pairs" in m:
        pairs = tuple(tuple(_str(s, f"{where}.pairs", _SYMBOL) for s in p) for p in _list(m["pairs"], f"{where}.pairs"))
        if any(len(p) != 2 for p in pairs):
            raise _fail(f"{where}.pairs", "each pair holds two symbols")
        symbols = tuple(s for p in pairs for s in p)
        by = _str(m.get("by"), f"{where}.by")
        if opts is None:
            raise _fail(where, "a pairs draw needs a candidate list")
        fams: dict[Any, list[str]] = {}
        for o in opts:
            fams.setdefault(ctx.candidates[m["options"]][o].get(by), []).append(o)
        if len(fams) != len(pairs) or any(len(v) != 2 for v in fams.values()):
            raise _fail(where, f"{len(pairs)} pairs need {len(pairs)} {by} values of exactly 2 options each")
    else:
        symbols = tuple(_str(s, f"{where}.symbols", _SYMBOL) for s in _list(m["symbols"], f"{where}.symbols"))
        by = None
        if opts is not None and len(symbols) != len(opts):
            raise _fail(where, f"{len(symbols)} symbols for {len(opts)} options")
    if len(set(symbols)) != len(symbols):
        raise _fail(where, "a symbol appears twice")
    pin = {}
    for o, syms in (m.get("pin") or {}).items():
        if opts is not None and o not in opts:
            raise _fail(f"{where}.pin.{o}", "is not one of the group's options")
        allowed = tuple(_list(syms, f"{where}.pin.{o}"))
        if not set(allowed) <= set(symbols):
            raise _fail(f"{where}.pin.{o}", f"{list(allowed)} are not all symbols of the group")
        pin[o] = allowed
    scope = tuple(m.get("slots") or ())
    if not set(scope) <= slot_ids:
        raise _fail(f"{where}.slots", f"{sorted(set(scope) - slot_ids)} are not slots of the stage")
    return Group(symbols, m["options"], filt, MappingProxyType(pin), pairs, by, scope)


def _slots(node: Any, where: str, sid: str, ctx: _Ctx) -> tuple[Slot, ...]:
    out, seen = [], set()
    for i, raw in enumerate(_list(node, where)):
        w = f"{where}[{i}]"
        m = _map(raw, w, ("id", "loop", "tiles"), ("kit",))
        slot_id = _str(m["id"], f"{w}.id")
        if not re.fullmatch(re.escape(sid) + r"\.[0-9]+", slot_id):
            raise _fail(f"{w}.id", f"{slot_id!r} is not {sid}.<n>")
        if slot_id in seen:
            raise _fail(f"{w}.id", f"{slot_id} appears twice")
        seen.add(slot_id)
        if m["loop"] not in ctx.loops:
            raise _fail(f"{w}.loop", f"{m['loop']!r} is not a loop of the plan")
        tiles = tuple(_list(m["tiles"], f"{w}.tiles"))
        if not 2 <= len(tiles) <= MAX_TILES or len(set(tiles)) != len(tiles):
            raise _fail(f"{w}.tiles", f"{list(tiles)}: 2-{MAX_TILES} distinct symbols")
        out.append(Slot(slot_id, m["loop"], tiles, m.get("kit")))
    return tuple(out)


def _check_tiles(st_id: str, slots: tuple[Slot, ...], groups: tuple[Group, ...], options: set[str]) -> None:
    for slot in slots:
        for sym in slot.tiles:
            if sym in options:
                raise _fail(f"stages.{st_id}.{slot.id}", f"{sym!r} is an option id, not a symbol: the plan places "
                                                         "only symbols on tiles (SHOOTOUTS.md section 2.3)")
            hits = [g for g in groups if sym in g.symbols and g.covers(slot.id)]
            if len(hits) != 1:
                raise _fail(f"stages.{st_id}.{slot.id}", f"{sym!r} is not a symbol of exactly one draw group "
                                                         f"({len(hits)} match)")


def _stage(sid: str, raw: Any, ctx: _Ctx) -> Stage:
    w = f"stages.{sid}"
    m = _map(raw, w, _STAGE_KEYS, _STAGE_OPT)
    day = m["day"]
    if day not in DAYS:
        raise _fail(f"{w}.day", f"{day!r} is not one of {list(DAYS)}")
    fills = m["fills"]
    if fills is not None and fills not in SLOT_LIST:
        raise _fail(f"{w}.fills", f"{fills!r} is not a band slot")
    bus = m["bus"]
    if bus not in BUSES:
        raise _fail(f"{w}.bus", f"{bus!r} is not one of {list(BUSES)}")
    sub = m.get("sub")
    if sub is not None and sub not in SUBS:
        raise _fail(f"{w}.sub", f"{sub!r} is not one of {list(SUBS)}")
    if fills is None and sub != "vamp":
        raise _fail(w, "a stage that fills no band slot substitutes the vamp line (sub: vamp)")
    if ("slots" in m) == ("slots_by_count" in m):
        raise _fail(w, "give either slots or slots_by_count")
    by_count: dict[int, tuple[Slot, ...]] = {}
    if "slots" in m:
        slots = _slots(m["slots"], f"{w}.slots", sid, ctx)
        all_ids = {s.id for s in slots}
    else:
        slots = ()
        for n, raw_slots in _map(m["slots_by_count"], f"{w}.slots_by_count").items():
            if not isinstance(n, int) or not 2 <= n <= MAX_TILES:
                raise _fail(f"{w}.slots_by_count", f"{n!r} is not an option count 2-{MAX_TILES}")
            by_count[n] = _slots(raw_slots, f"{w}.slots_by_count.{n}", sid, ctx)
        all_ids = {s.id for v in by_count.values() for s in v}
    groups = tuple(_group(g, f"{w}.draw[{i}]", ctx, all_ids) for i, g in enumerate(_list(m["draw"], f"{w}.draw")))
    if sub == "vamp" and any(g.options != "vamps" for g in groups):
        raise _fail(f"{w}.draw", "a vamp stage draws from the vamps list")
    for g in groups:
        if fills and g.leaders and g.leaders != fills:
            raise _fail(f"{w}.draw", f"draws leaders of {g.leaders} but fills {fills}")
        if fills and not g.leaders and g.options != SLOT_LIST[fills]:
            raise _fail(f"{w}.draw", f"draws {g.options} but fills {fills} ({SLOT_LIST[fills]})")
    for slots_ in [slots, *by_count.values()]:
        _check_tiles(sid, slots_, groups, set(ctx.owner))
    for n, slots_ in by_count.items():
        if any(len(s.tiles) != n for s in slots_):
            raise _fail(f"{w}.slots_by_count.{n}", "every slot holds that many tiles")
        if any(g.leaders and len(g.symbols) < n for g in groups):
            raise _fail(f"{w}.slots_by_count.{n}", "more options than the group has symbols")
    band = ctx.band(m.get("band") or {}, f"{w}.band")
    default = m.get("default")
    if default is not None and not any(default in (ctx.candidates.get(g.options) or {}) for g in groups):
        raise _fail(f"{w}.default", f"{default!r} is not one of the stage's options")
    sax_muted = bool(m.get("sax_muted", False))
    return Stage(id=sid, day=day, category=_str(m["category"], f"{w}.category"), fills=fills, bus=bus,
                 what=m.get("what"), groups=groups, slots=slots, slots_by_count=MappingProxyType(by_count),
                 band=MappingProxyType(band), default=default, sub=sub, sax_muted=sax_muted,
                 anchor=bool(m.get("anchor", not sax_muted)), repeat_group=m.get("repeat_group"))


def _batch(raw: Any, where: str, stages: Mapping[str, Stage]) -> Batch:
    m = _map(raw, where, ("name", "date", "day", "category", "note", "stages"), ("anchor", "split"))
    name = m["name"]
    if not isinstance(name, str) or not BATCH_RE.fullmatch(name):
        raise _fail(f"{where}.name", f"batch name {name!r} must match {BATCH_RE.pattern}")
    _str(str(m["date"]), f"{where}.date", _DATE)
    if m["day"] not in DAYS:
        raise _fail(f"{where}.day", f"{m['day']!r} is not one of {list(DAYS)}")
    sids = tuple(_list(m["stages"], f"{where}.stages"))
    for s in sids:
        if s not in stages:
            raise _fail(f"{where}.stages", f"{s!r} is not a stage")
        if stages[s].day != m["day"]:
            raise _fail(f"{where}.stages", f"stage {s} belongs to {stages[s].day}, not {m['day']}")
    return Batch(name, str(m["date"]), m["day"], _str(m["category"], f"{where}.category"),
                 _str(m["note"], f"{where}.note"), sids, bool(m.get("anchor", False)), m.get("split"))


def _cut(name: str, raw: Any, stages: Mapping[str, Stage]) -> Cut:
    w = f"cuts.{name}"
    m = _map(raw, w, ("day", "edits"))
    edits = []
    for i, e in enumerate(_list(m["edits"], f"{w}.edits")):
        ew = f"{w}.edits[{i}]"
        em = _map(e, ew, ("stage",), ("keep", "tiles", "pin", "drop"))
        st = stages.get(em["stage"])
        if st is None:
            raise _fail(f"{ew}.stage", f"{em['stage']!r} is not a stage")
        ids = {s.id for s in st.slots}
        keep = tuple(em["keep"]) if "keep" in em else None
        if keep is not None and not set(keep) <= ids:
            raise _fail(f"{ew}.keep", f"{sorted(set(keep) - ids)} are not slots of {st.id}")
        tiles = {}
        for slot_id, t in (em.get("tiles") or {}).items():
            if slot_id not in ids:
                raise _fail(f"{ew}.tiles", f"{slot_id!r} is not a slot of {st.id}")
            if set(t) != set(next(s for s in st.slots if s.id == slot_id).tiles):
                raise _fail(f"{ew}.tiles.{slot_id}", "must hold the slot's own symbols")
            tiles[slot_id] = tuple(t)
        pin = {}
        for o, syms in (em.get("pin") or {}).items():
            if not set(syms) <= st.symbols:
                raise _fail(f"{ew}.pin.{o}", f"{syms} are not symbols of {st.id}")
            pin[o] = tuple(syms)
        edits.append(CutEdit(st.id, keep, MappingProxyType(tiles), MappingProxyType(pin), bool(em.get("drop"))))
    return Cut(name, m["day"], tuple(edits))


# ------------------------------------------------------------------------------------------------ cuts
def apply_cuts(plan: Plan, names: list[str] | tuple[str, ...]) -> Plan:
    """The plan with the named cuts of section 2.7 applied (a new Plan; `plan` is unchanged)."""
    stages = dict(plan.stages)
    for name in names:
        cut = plan.cuts.get(name)
        if cut is None:
            raise PlanError(f"unknown cut {name!r}; the plan has {sorted(plan.cuts)}")
        for e in cut.edits:
            st = stages.get(e.stage)
            if st is None:
                continue
            if e.drop:
                del stages[e.stage]
                continue
            slots = st.slots if e.keep is None else tuple(s for s in st.slots if s.id in e.keep)
            slots = tuple(replace(s, tiles=e.tiles.get(s.id, s.tiles)) for s in slots)
            groups = st.groups
            if e.pin:
                groups = tuple(replace(g, pin=MappingProxyType({**g.pin, **{o: v for o, v in e.pin.items()
                                                                            if set(v) <= set(g.symbols)}}))
                               for g in groups)
            stages[e.stage] = replace(st, slots=slots, groups=groups)
    batches = tuple(replace(b, stages=tuple(s for s in b.stages if s in stages)) for b in plan.batches)
    return replace(plan, stages=MappingProxyType(stages), batches=batches,
                   applied_cuts=plan.applied_cuts + tuple(names))
