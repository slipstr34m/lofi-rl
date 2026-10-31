"""shootouts/plan.yaml: the shootout plan in symbols (docs/SHOOTOUTS.md §4, §6; §7.8 "Plan").

`load_plan` validates the file strictly, as palette.py does a palette: unknown keys are refused, every reference
(a slot's loop and tiles, a stage's options and band, the day's band and leaders, a batch's stages, a cut's slots,
the level anchors, the repeat groups) must resolve, and a tile may hold only a symbol of its stage, never an option
(§2.3: the plan is public). Paths in the plan are relative to the repo root.

Every reader takes the same Plan: the palette generator (§7.8) and the level tool (§7.1) read the candidates, the
stages' options and the day's band; the job-time draw (§2.3) reads each stage's draw groups (`Stage.groups`, one per
draw: K1's family pairs, S1's two registers, the vamps' one per mood, a ranks stage's leaders), the band slots
(`Day.band_slots`, `Day.leaders`), the batches, the repeat groups and the S2 re-run. `apply_cuts` gives the plan with
§2.7's cuts applied, and `substitute` rewrites a loop's statement lines (§5 Substitutions).

    from lofi_rl.shootout.plan import apply_cuts, load_plan
    plan = apply_cuts(load_plan(), ["latin3"])
    plan.stages["K1"].slots[0].tiles        # ("P1a", "P1b", "P2a", "P2b")

Standard library and PyYAML only.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from lofi_rl.brief import SCENE, Brief
from lofi_rl.palette import ROOT
from lofi_rl.ui.manifest import BATCH_RE

PLAN = ROOT / "shootouts" / "plan.yaml"
BUSES = ("keys", "drums", "bass", "sax")                 # the instrument buses a candidate plays
STAGE_BUSES = BUSES + ("bed", "keys+bass")               # what a round's hidden `bus_lufs` reads (§2.2)
DRAW_RULES = ("any", "family_pairs", "registers", "central", "ranks")
DAYS = ("day1", "day2", "day3")                          # the job's --stage names of days 1-3
# The band: every slot a clip's palette fills (§4 "fixed in every round", CD-7), and the plain name of the list
# its options come from (the job's error messages).
SLOT_LIST: Mapping[str, str] = MappingProxyType({
    "keys": "keys", "kit": "kits", "bass": "basses", "sax": "saxes", "swing": "swing", "humanize": "humanize",
    "sax_variant": "sax_variants", "timing": "timing", "bed": "excerpts", "rain": "rain"})
BAND_SLOTS = tuple(SLOT_LIST)
SLOT_BUS: Mapping[str, str] = MappingProxyType({"keys": "keys", "kit": "drums", "bass": "bass", "sax": "sax"})
BUS_SLOT: Mapping[str, str] = MappingProxyType({v: k for k, v in SLOT_BUS.items()})
MAX_TILES = 4                                            # the page's limit (RATING.md: 2-4 clips a shootout round)
_ID = re.compile(r"[a-z][a-z0-9_]*")
_STAGE = re.compile(r"[A-Z][A-Za-z0-9_]*")
_SLOT = re.compile(r"[A-Z][A-Za-z0-9_]*\.\d+")
_LOOP = re.compile(r"L[1-9]")
_SHA = re.compile(r"[0-9a-f]{64}")
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)     # libyaml when present; the same documents


class PlanError(ValueError):
    pass


def day_number(day: int | str) -> int:
    """1, 2 or 3 from a day number or the job's `day1`..`day3`."""
    if isinstance(day, str) and day in DAYS:
        return DAYS.index(day) + 1
    if isinstance(day, int) and not isinstance(day, bool) and 1 <= day <= 3:
        return day
    raise PlanError(f"{day!r} is not one of {list(DAYS)}")


def day_name(day: int | str) -> str:
    return DAYS[day_number(day) - 1]


# ----------------------------------------------------------------------------------------------- schema
@dataclass(frozen=True)
class Candidate:
    id: str
    bus: str                                 # keys | drums | bass | sax
    source: str | None = None                # a source id of the base palette (SoundFont instruments)
    bank: int | None = None
    prog: int | None = None
    family: str | None = None                # keys: the K1 twin pair (CD-8)
    register: str | None = None              # sax: tenor | alto
    archive: Mapping[str, Any] | None = field(default=None, hash=False)   # a §7.4 kit: path, blocks, sha256, ...
    base: bool = False                       # a kit that is the base palette's own `dusty` lanes


@dataclass(frozen=True)
class Variant:
    id: str
    stage: str
    set: Mapping[str, Any] = field(hash=False)
    what: str = ""


@dataclass(frozen=True)
class Vamp:
    id: str
    mood: str
    chords: str                              # "Im9 bVImaj7"
    span: int
    margin: int
    borrowed: bool

    @property
    def line(self) -> str:
        """The `vamp:` line's value: `Im9 bVImaj7 span 2`."""
        return f"{self.chords} span {self.span}"


@dataclass(frozen=True)
class Loop:
    id: str                                  # L1 .. L9
    name: str                                # the file stem, e.g. l1_melancholy_low; the hidden manifests' brief_id
    path: Path
    brief: Brief

    @property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class Slot:
    id: str                                  # "K1.1"
    stage: str
    loop: str                                # L1..L9
    tiles: tuple[str, ...]                   # symbols, tile 1 first
    kit: str | None = None                   # R2: the symbol of the day's kit stage whose kit plays in this round


@dataclass(frozen=True)
class Group:
    """One draw of a stage: its symbols stand for its options, one each, drawn at job time within `rule`."""

    symbols: tuple[str, ...]
    options: tuple[str, ...]                 # () when they are the job's leaders (`leaders`)
    rule: str                                # any | family_pairs | central | ranks (a register group is `any`)
    slots: tuple[str, ...] = ()              # the slots it draws for (empty: every slot of the stage)
    leaders: str | None = None               # a ranks group: the band slot whose --leaders it maps
    pin: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}), hash=False)
    pairs: tuple[tuple[str, str], ...] = ()  # family_pairs: the twin symbols, in plan order
    by: str | None = None                    # family_pairs: the candidate field a pair shares
    where: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), hash=False)  # what its options share

    def covers(self, slot: str) -> bool:
        return not self.slots or slot in self.slots


@dataclass(frozen=True)
class Stage:
    id: str
    day: int
    category: str
    symbols: tuple[str, ...]
    options: tuple[str, ...]                 # () for a ranks stage: its options come from a tally (§8)
    draw: Mapping[str, Any] | None = field(default=None, hash=False)   # None for a grouped stage (the vamps)
    slots: tuple[Slot, ...] = ()
    bus: str | None = None
    what: str = ""
    sub: Mapping[str, str] = field(default_factory=dict, hash=False)
    ranks: Mapping[str, Any] | None = field(default=None, hash=False)
    tie: Mapping[str, Any] | None = field(default=None, hash=False)
    fills: str | None = None                 # the band slot the option takes; None: the option is a loop line
    default: str | None = None
    band: Mapping[str, str | None] = field(default_factory=dict, hash=False)
    sax_muted: bool = False
    repeat_group: str | None = None
    groups: tuple[Group, ...] = ()

    @property
    def anchor(self) -> bool:
        """An anchor may imitate this stage's rounds: they play a sax (§2.5)."""
        return not self.sax_muted

    @property
    def day_name(self) -> str:
        return day_name(self.day)

    @property
    def slots_by_count(self) -> Mapping[int, tuple[Slot, ...]]:
        """A tie stage's slots by the number of options it plays (CD-41, §4.4); empty for any other stage."""
        if self.tie is None:
            return MappingProxyType({})
        tied = tuple(replace(s, tiles=tuple(t)) for s, t in zip(self.slots, self.tie["tiles"]))
        return MappingProxyType({len(self.symbols): self.slots, len(self.tie["symbols"]): tied})

    def slots_for(self, count: int | None = None) -> tuple[Slot, ...]:
        """The round slots; for a stage drawn from the job's leaders, those for `count` options."""
        by = self.slots_by_count
        if not by:
            return self.slots
        if count not in by:
            raise PlanError(f"stage {self.id} has slots for {sorted(by)} options, not {count}")
        return by[count]

    def group_of(self, slot: str, symbol: str) -> int:
        """Index of the group that resolves `symbol` in `slot` (exactly one, checked at load)."""
        hits = [i for i, g in enumerate(self.groups) if symbol in g.symbols and g.covers(slot)]
        if len(hits) != 1:
            raise PlanError(f"stage {self.id} slot {slot}: symbol {symbol!r} resolves to {len(hits)} groups")
        return hits[0]


@dataclass(frozen=True)
class Day:
    day: int
    date: str
    stages: tuple[str, ...]
    band: Mapping[str, str] = field(hash=False)     # bus -> candidate id (Day 1) or a tally phrase (Days 2-3)
    leaders: Mapping[str, str | None] = field(default_factory=dict, hash=False)  # slot -> default, None: the job names it
    fixed: Mapping[str, str | None] = field(default_factory=dict, hash=False)    # the other band slots

    def band_slots(self) -> dict[str, str | None]:
        """Every band slot: the fixed ones, the Day-1 band's instruments under their slot names (drums -> kit), and
        None for the slots the job's --leaders fills."""
        out: dict[str, str | None] = {slot: None for slot in BAND_SLOTS}
        if not self.leaders:
            out.update({BUS_SLOT[bus]: inst for bus, inst in self.band.items()})
        out.update(self.fixed)
        return out


@dataclass(frozen=True)
class Batch:
    names: tuple[str, ...]
    date: str
    day: int
    rounds: Mapping[str, int] = field(hash=False)
    anchors: int = 0
    category: str = ""
    note: str = ""
    est_min_plan: float = 0.0
    est_min_full: float = 0.0
    split: str | None = None

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(self.rounds)


@dataclass(frozen=True)
class Cut:
    id: str
    day: int
    order: int
    drop: tuple[str, ...]
    retile: Mapping[str, list] = field(default_factory=dict, hash=False)
    default_on: Mapping[str, list] = field(default_factory=dict, hash=False)
    saves_min: tuple[float, ...] = ()


@dataclass(frozen=True)
class Levels:
    anchors: Mapping[str, str] = field(hash=False)  # bus -> the GU instrument g of that bus (§3 step 2)
    sax_target: str = ""                            # S* = the median of this instrument's L (§3 step 4)
    zero_trim_source: str = ""                      # "GU instruments get t = 0"
    linear_drop_db: float = 6.02
    linear_tol_db: float = 0.1
    median_tol_lu: float = 0.05
    loop_tol_lu: float = 0.1


@dataclass(frozen=True)
class Plan:
    path: Path
    sha256: str
    root: Path
    base: str                                       # repo-relative path of the base palette
    loops_dir: Path
    loop_names: Mapping[str, str] = field(hash=False)
    voices: Mapping[str, str] = field(hash=False)
    registers: Mapping[str, Mapping[str, list]] = field(hash=False)
    candidates: Mapping[str, Candidate] = field(hash=False)
    variants: Mapping[str, Variant] = field(hash=False)
    vamps: Mapping[str, Vamp] = field(hash=False)
    stages: Mapping[str, Stage] = field(hash=False)
    days: Mapping[int, Day] = field(hash=False)
    batches: tuple[Batch, ...] = ()
    controls: Mapping[str, Any] = field(default_factory=dict, hash=False)
    cuts: tuple[Cut, ...] = ()
    levels: Levels = field(default_factory=lambda: Levels({}))
    scene: str = SCENE
    alto_above: frozenset[str] = frozenset()
    loops: Mapping[str, Loop] = field(default_factory=dict, hash=False)
    band_options: Mapping[str, tuple[str, ...]] = field(default_factory=dict, hash=False)   # slot -> its options
    rerun_day: int | None = None
    rerun_stages: tuple[str, ...] = ()
    applied_cuts: tuple[str, ...] = ()

    def briefs(self) -> dict[str, Brief]:
        """loop name -> its brief, from <loops>/briefs.yaml (§5)."""
        return read_briefs(self.loops_dir)

    def loop_file(self, name: str) -> Path:
        return self.loops_dir / f"{self.loop_names[name]}.ls"

    def loop_text(self, name: str) -> str:
        """The text of loop L1..L9 as committed."""
        return self.loop_file(name).read_text(encoding="utf-8")

    def loop_named(self, name: str) -> Loop:
        for lp in self.loops.values():
            if lp.name == name:
                return lp
        raise PlanError(f"no loop is named {name!r}")

    def stages_of(self, day: int | str) -> tuple[Stage, ...]:
        """The day's stages in plan order (after cuts, those left)."""
        n = day_number(day)
        return tuple(self.stages[s] for s in self.days[n].stages if s in self.stages)

    def batches_of(self, day: int | str) -> tuple[Batch, ...]:
        n = day_number(day)
        return tuple(b for b in self.batches if b.day == n)

    def cut(self, cut_id: str) -> Cut:
        for c in self.cuts:
            if c.id == cut_id:
                return c
        raise PlanError(f"unknown cut {cut_id!r}; the plan has {sorted(c.id for c in self.cuts)}")

    @property
    def repeat_day(self) -> int:
        return self.controls["repeats"]["day"]

    @property
    def repeat_groups(self) -> Mapping[str, tuple[str, ...]]:
        """group -> its stages (§2.5: one repeat per group)."""
        return MappingProxyType({g: tuple(s) for g, s in self.controls["repeats"]["groups"].items()})


def read_briefs(loops_dir: Path) -> dict[str, Brief]:
    doc = yaml.load((Path(loops_dir) / "briefs.yaml").read_text(encoding="utf-8"), Loader=_LOADER)
    if not isinstance(doc, dict) or not isinstance(doc.get("loops"), dict):
        raise PlanError(f"{loops_dir}/briefs.yaml: expected a mapping with `loops`")
    out = {}
    for name, b in doc["loops"].items():
        try:
            out[str(name)] = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"])
        except (KeyError, TypeError, ValueError) as err:
            raise PlanError(f"{loops_dir}/briefs.yaml: loop {name}: {err}") from err
    return out


def substitute(text: str, **lines: str) -> str:
    """The loop text with each named statement's line replaced: substitute(t, keys="ep spread") turns
    `keys: ep rootless` into `keys: ep spread`, and vamp="Im9 IVm9 span 2" replaces the `vamp:` line (§5
    Substitutions). Each statement must appear exactly once, at the start of a line; every other byte is kept."""
    out = text.splitlines(keepends=True)
    for head, value in lines.items():
        hits = [i for i, line in enumerate(out) if line.startswith(f"{head}:")]
        if len(hits) != 1:
            raise PlanError(f"substitute: the loop has no '{head}:' line" if not hits
                            else f"substitute: the loop has {len(hits)} '{head}:' lines")
        end = "\n" if out[hits[0]].endswith("\n") else ""
        out[hits[0]] = f"{head}: {value}{end}"
    return "".join(out)


# ---------------------------------------------------------------------------------------------- loading
def load_plan(path: Path | str = PLAN, root: Path = ROOT) -> Plan:
    """Load and validate a plan file. Raises PlanError naming the first problem."""
    path = Path(path)
    data = path.read_bytes()
    try:
        doc = yaml.load(data.decode("utf-8"), Loader=_LOADER)
    except (UnicodeDecodeError, yaml.YAMLError) as err:
        raise PlanError(f"{path}: not a UTF-8 YAML document ({err})") from err
    return parse_plan(doc, path, hashlib.sha256(data).hexdigest(), root=root)


def _fail(where: str, what: str) -> PlanError:
    return PlanError(f"{where}: {what}")


def _map(node: Any, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(node, dict):
        raise _fail(where, f"expected a mapping, got {type(node).__name__}")
    keys = {str(k) for k in node}
    missing = [k for k in required if k not in keys]
    unknown = sorted(keys - set(required) - set(optional))
    if missing:
        raise _fail(where, "missing " + ", ".join(missing))
    if unknown:
        raise _fail(where, "unknown key(s) " + ", ".join(unknown))
    return {str(k): v for k, v in node.items()}


def _str(node: Any, where: str, pattern: re.Pattern | None = None) -> str:
    if isinstance(node, (_dt.date, _dt.datetime)):
        node = node.isoformat()
    if not isinstance(node, str) or not node.strip():
        raise _fail(where, f"expected a non-empty string, got {node!r}")
    if pattern is not None and not pattern.fullmatch(node):
        raise _fail(where, f"{node!r} does not match {pattern.pattern}")
    return node


def _int(node: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(node, bool) or not isinstance(node, int) or not lo <= node <= hi:
        raise _fail(where, f"expected an integer in {lo}..{hi}, got {node!r}")
    return node


def _num(node: Any, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise _fail(where, f"expected a number, got {node!r}")
    return float(node)


def _strs(node: Any, where: str, pattern: re.Pattern | None = None, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(node, list) or not node:
        raise _fail(where, "expected a non-empty list")
    out = tuple(_str(v, f"{where}[{i}]", pattern) for i, v in enumerate(node))
    if unique and len(set(out)) != len(out):
        raise _fail(where, f"lists a value twice: {list(out)}")
    return out


def parse_plan(doc: Any, path: Path | str = "", sha256: str = "", *, root: Path = ROOT) -> Plan:
    top = _map(doc, "plan", ("version", "scene", "base", "loops", "loop_names", "voices", "registers", "candidates",
                             "variants", "vamps", "stages", "days", "batches", "controls", "cuts", "levels"),
               ("alto_above", "rerun_s2"))
    if top["version"] != 1:
        raise _fail("version", f"expected 1, got {top['version']!r}")
    if top["scene"] != SCENE:
        raise _fail("scene", f"must be {SCENE!r} (the only scene, TASTE.md)")
    loops_dir = root / _str(top["loops"], "loops")
    names = _map(top["loop_names"], "loop_names", (), tuple(f"L{i}" for i in range(1, 10)))
    if not names:
        raise _fail("loop_names", "names no loop (the plan's are L1..L9, §5)")
    loop_names = {k: _str(v, f"loop_names.{k}", _ID) for k, v in names.items()}
    loops = _loops(loops_dir, loop_names)
    alto_above = frozenset(_strs(top["alto_above"], "alto_above") if top.get("alto_above") is not None else ())
    for lp in alto_above:
        if lp not in loop_names:
            raise _fail("alto_above", f"{lp!r} is not one of L1..L9")
    voices = _map(top["voices"], "voices", ("keys", "kit"))
    registers = {}
    for reg, r in _map(top["registers"], "registers", ("tenor", "alto")).items():
        m = _map(r, f"registers.{reg}", ("play", "comfort", "home"))
        registers[reg] = MappingProxyType({k: [_int(x, f"registers.{reg}.{k}", 0, 127) for x in m[k]] for k in m})
    candidates = _candidates(top["candidates"])
    variants = _variants(top["variants"])
    vamps = _vamps(top["vamps"])
    seen = [*candidates, *variants, *vamps]
    twice = sorted({o for o in seen if seen.count(o) > 1})
    if twice:
        raise _fail("plan", f"{twice} appear twice among the candidates, variants and vamps: an option id is unique")
    stages = _stages(top["stages"], loop_names, loops, candidates, variants, vamps)
    band_options = _band_options(candidates, variants, stages)
    for st in stages.values():
        for slot, opt in st.band.items():
            if opt is not None and opt not in band_options[slot]:
                raise _fail(f"stages.{st.id}.band.{slot}", f"{opt!r} is not one of the {SLOT_LIST[slot]} options")
    days = _days(top["days"], stages, candidates, band_options)
    batches = _batches(top["batches"], stages)
    cuts = _cuts(top["cuts"], stages)
    levels = _levels(top["levels"], candidates)
    controls = _map(top["controls"], "controls", ("repeats", "anchors"))
    rep = _map(controls["repeats"], "controls.repeats", ("count", "groups", "day", "min_days_after", "tiles", "never"))
    _int(rep["day"], "controls.repeats.day", 1, 3)
    groups_of: dict[str, str] = {}
    for group, members in _map(rep["groups"], "controls.repeats.groups",
                               tuple(rep["groups"]) if isinstance(rep["groups"], dict) else ()).items():
        for s in _strs(members, f"controls.repeats.groups.{group}"):
            if s not in stages:
                raise _fail(f"controls.repeats.groups.{group}", f"{s!r} is not a stage")
            if s in groups_of:
                raise _fail(f"controls.repeats.groups.{group}", f"{s} is already in group {groups_of[s]}")
            groups_of[s] = group
    stages = {sid: replace(st, repeat_group=groups_of.get(sid)) for sid, st in stages.items()}
    _map(controls["anchors"], "controls.anchors", ("count", "per_day", "sour", "gate"))
    rerun_day, rerun_stages = None, ()
    if top.get("rerun_s2") is not None:
        rr = _map(top["rerun_s2"], "rerun_s2", ("day", "stages"))
        rerun_day = _int(rr["day"], "rerun_s2.day", 1, 3)
        rerun_stages = _strs(rr["stages"], "rerun_s2.stages")
        for s in rerun_stages:
            if s not in stages:
                raise _fail("rerun_s2.stages", f"{s!r} is not a stage")
    for b in batches:
        if b.split is not None and len(b.names) < 2:
            raise _fail(f"batches {list(b.names)}", "a split pool names two batches or more")
        if b.split is None and len(b.names) != 1:
            raise _fail(f"batches {list(b.names)}", "a row naming several batches shares a `split` pool")
    every = [n for b in batches for n in b.names]
    if len(set(every)) != len(every):
        raise _fail("batches", "a batch name appears twice")
    return Plan(path=Path(path), sha256=sha256, root=root, base=_str(top["base"], "base"), loops_dir=loops_dir,
                loop_names=MappingProxyType(loop_names), voices=MappingProxyType(dict(voices)),
                registers=MappingProxyType(registers), candidates=MappingProxyType(candidates),
                variants=MappingProxyType(variants), vamps=MappingProxyType(vamps),
                stages=MappingProxyType(stages), days=MappingProxyType(days), batches=batches,
                controls=MappingProxyType(controls), cuts=cuts, levels=levels, scene=top["scene"],
                alto_above=alto_above, loops=MappingProxyType(loops),
                band_options=MappingProxyType(band_options), rerun_day=rerun_day, rerun_stages=rerun_stages)


def _loops(loops_dir: Path, loop_names: Mapping[str, str]) -> dict[str, Loop]:
    """Every L1..L9 with its file and brief: a missing file or brief is refused at load."""
    try:
        briefs = read_briefs(loops_dir)
    except OSError as err:
        raise _fail("loops", f"cannot read {loops_dir}/briefs.yaml ({err})") from err
    out = {}
    for lid, name in loop_names.items():
        file = loops_dir / f"{name}.ls"
        if not file.is_file():
            raise _fail(f"loop_names.{lid}", f"{file} is missing")
        if name not in briefs:
            raise _fail(f"loop_names.{lid}", f"{name} has no brief in {loops_dir}/briefs.yaml")
        out[lid] = Loop(lid, name, file, briefs[name])
    if len({lp.name for lp in out.values()}) != len(out):
        raise _fail("loop_names", "a loop is listed twice")
    return out


def _candidates(node: Any) -> dict[str, Candidate]:
    if not isinstance(node, dict) or not node:
        raise _fail("candidates", "expected a non-empty mapping")
    out = {}
    for cid, raw in node.items():
        w = f"candidates.{cid}"
        _str(cid, f"{w} id", _ID)
        bus = _str((raw or {}).get("bus") if isinstance(raw, dict) else None, f"{w}.bus")
        if bus not in BUSES:
            raise _fail(f"{w}.bus", f"{bus!r} is not one of {list(BUSES)}")
        if bus == "drums":
            m = _map(raw, w, ("bus",), ("archive", "base"))
            if ("archive" in m) == bool(m.get("base")):
                raise _fail(w, "a kit is either a §7.4 archive or the base palette's own (base: true)")
            archive = None
            if "archive" in m:
                a = _map(m["archive"], f"{w}.archive", ("path", "blocks", "sha256", "kit_map", "license"), ("mics",))
                for k in ("path", "blocks", "kit_map", "license"):
                    _str(a[k], f"{w}.archive.{k}")
                if a["sha256"] is not None:
                    _str(a["sha256"], f"{w}.archive.sha256", _SHA)
                archive = MappingProxyType(dict(a))
            out[cid] = Candidate(cid, bus, archive=archive, base=bool(m.get("base")))
            continue
        extra = {"keys": ("family",), "sax": ("register",), "bass": ()}[bus]
        m = _map(raw, w, ("bus", "source", "bank", "prog") + extra)
        register = _str(m["register"], f"{w}.register") if bus == "sax" else None
        if register is not None and register not in ("tenor", "alto"):
            raise _fail(f"{w}.register", f"{register!r} is not tenor or alto")
        out[cid] = Candidate(cid, bus, source=_str(m["source"], f"{w}.source", _ID),
                             bank=_int(m["bank"], f"{w}.bank", 0, 128), prog=_int(m["prog"], f"{w}.prog", 0, 127),
                             family=_str(m["family"], f"{w}.family", _ID) if bus == "keys" else None,
                             register=register)
    return out


def _variants(node: Any) -> dict[str, Variant]:
    out = {}
    for vid, raw in (node or {}).items():
        w = f"variants.{vid}"
        _str(vid, f"{w} id", _ID)
        m = _map(raw, w, ("stage", "set", "what"))
        if not isinstance(m["set"], dict):
            raise _fail(f"{w}.set", "expected a mapping of dotted palette paths")
        out[vid] = Variant(vid, _str(m["stage"], f"{w}.stage", _STAGE), MappingProxyType(dict(m["set"])),
                           _str(m["what"], f"{w}.what"))
    return out


def _vamps(node: Any) -> dict[str, Vamp]:
    out = {}
    for vid, raw in (node or {}).items():
        w = f"vamps.{vid}"
        _str(vid, f"{w} id", _ID)
        m = _map(raw, w, ("mood", "chords", "span", "margin", "borrowed"))
        mood = _str(m["mood"], f"{w}.mood")
        if mood not in ("melancholy", "wistful", "hopeful"):
            raise _fail(f"{w}.mood", f"{mood!r} is not a mood")
        chords = _str(m["chords"], f"{w}.chords")
        if len(chords.split()) != 2:
            raise _fail(f"{w}.chords", "a vamp is two chords")
        if not isinstance(m["borrowed"], bool):
            raise _fail(f"{w}.borrowed", "expected true or false")
        out[vid] = Vamp(vid, mood, chords, _int(m["span"], f"{w}.span", 1, 2), _int(m["margin"], f"{w}.margin", 0, 99),
                        m["borrowed"])
    return out


def _band_options(candidates: Mapping[str, Candidate], variants: Mapping[str, Variant],
                  stages: Mapping[str, Stage]) -> dict[str, tuple[str, ...]]:
    """Each band slot's options: the candidates of its bus, or the variants of the stage that fills it."""
    out = {}
    for slot in BAND_SLOTS:
        if slot in SLOT_BUS:
            out[slot] = tuple(c for c, x in candidates.items() if x.bus == SLOT_BUS[slot])
        else:
            out[slot] = tuple(v for v, x in variants.items() if x.stage in stages and stages[x.stage].fills == slot)
    return out


def _check_options(options: tuple[str, ...], where: str, sid: str, bus: str | None, fills: str | None,
                   candidates: Mapping[str, Candidate], variants: Mapping[str, Variant],
                   vamps: Mapping[str, Vamp]) -> None:
    for o in options:
        if o in candidates:
            if bus is None or candidates[o].bus != bus:
                raise _fail(where, f"candidate {o!r} plays bus {candidates[o].bus}, not {bus}")
            if fills is not None and SLOT_BUS.get(fills) != candidates[o].bus:
                raise _fail(where, f"candidate {o!r} plays bus {candidates[o].bus}, but the stage fills {fills}")
        elif o in variants:
            if variants[o].stage != sid:
                raise _fail(where, f"variant {o!r} belongs to stage {variants[o].stage}")
            if fills is None or fills in SLOT_BUS:
                raise _fail(where, f"variant {o!r} needs a stage that fills a variant slot, not {fills}")
        elif o in vamps:
            if fills is not None:
                raise _fail(where, f"vamp {o!r} is a loop line: its stage fills no band slot (fills: null)")
        else:
            raise _fail(where, f"{o!r} is not a candidate, variant or vamp")


def _stages(node: Any, loop_names: Mapping[str, str], loops: Mapping[str, Loop],
            candidates: Mapping[str, Candidate], variants: Mapping[str, Variant],
            vamps: Mapping[str, Vamp]) -> dict[str, Stage]:
    if not isinstance(node, dict) or not node:
        raise _fail("stages", "expected a non-empty mapping")
    out: dict[str, Stage] = {}
    for sid, raw in node.items():
        w = f"stages.{sid}"
        _str(sid, f"{w} id", _STAGE)
        m = _map(raw, w, ("day", "bus", "fills", "category", "symbols", "slots"),
                 ("what", "options", "draw", "groups", "ranks", "sub", "tie", "default", "band", "sax_muted"))
        symbols = _strs(m["symbols"], f"{w}.symbols")
        bus = _str(m["bus"], f"{w}.bus")
        if bus not in STAGE_BUSES:
            raise _fail(f"{w}.bus", f"{bus!r} is not one of {list(STAGE_BUSES)}")
        fills = m["fills"]
        if fills is not None and fills not in BAND_SLOTS:
            raise _fail(f"{w}.fills", f"{fills!r} is not a band slot {list(BAND_SLOTS)}")
        if fills in SLOT_BUS and SLOT_BUS[fills] != bus:
            raise _fail(f"{w}.fills", f"a stage that fills {fills} plays bus {SLOT_BUS[fills]}, not {bus}")
        ways = [k for k in ("options", "ranks", "groups") if k in m]
        if len(ways) != 1:
            raise _fail(w, "a stage names its options, the ranks of an earlier stage, or draw groups: exactly one")
        if ("draw" in m) == ("groups" in m):
            raise _fail(w, "a stage has one `draw`, or `groups` each with its own")
        options: tuple[str, ...] = ()
        ranks = None
        draw = None
        group_specs: list[tuple[tuple[str, ...], tuple[str, ...], Mapping[str, Any], str]] = []
        if "groups" in m:
            if not isinstance(m["groups"], list) or not m["groups"]:
                raise _fail(f"{w}.groups", "expected a non-empty list")
            for i, g in enumerate(m["groups"]):
                gw = f"{w}.groups[{i}]"
                gm = _map(g, gw, ("slots", "options", "draw"))
                g_opts = _strs(gm["options"], f"{gw}.options")
                _check_options(g_opts, f"{gw}.options", sid, bus, fills, candidates, variants, vamps)
                if len(g_opts) != len(symbols):
                    raise _fail(gw, f"{len(symbols)} symbols for {len(g_opts)} options")
                g_draw = _draw(gm["draw"], f"{gw}.draw", symbols, g_opts)
                if g_draw["rule"] not in ("any", "central"):
                    raise _fail(f"{gw}.draw", "a draw group's rule is any or central")
                group_specs.append((_strs(gm["slots"], f"{gw}.slots", _SLOT), g_opts, g_draw, gw))
                options += g_opts
            if len(set(options)) != len(options):
                raise _fail(f"{w}.groups", "an option is in two groups")
        elif "options" in m:
            options = _strs(m["options"], f"{w}.options")
            _check_options(options, f"{w}.options", sid, bus, fills, candidates, variants, vamps)
            if len(options) != len(symbols):
                raise _fail(w, f"{len(symbols)} symbols for {len(options)} options")
            draw = _draw(m["draw"], f"{w}.draw", symbols, options)
        else:
            r = _map(m["ranks"], f"{w}.ranks", ("of", "top"))
            ranks = MappingProxyType({"of": _str(r["of"], f"{w}.ranks.of", _STAGE),
                                      "top": _int(r["top"], f"{w}.ranks.top", 1, 4)})
            draw = _draw(m["draw"], f"{w}.draw", symbols, options)
            if draw["rule"] != "ranks":
                raise _fail(f"{w}.draw", "a ranks stage draws by `rule: ranks`")
        if draw is not None and draw["rule"] == "ranks" and ranks is None:
            raise _fail(f"{w}.draw", "`rule: ranks` needs the stage's `ranks`")
        slots = []
        if not isinstance(m["slots"], list) or not m["slots"]:
            raise _fail(f"{w}.slots", "expected a non-empty list")
        for i, s in enumerate(m["slots"]):
            sw = f"{w}.slots[{i}]"
            sm = _map(s, sw, ("id", "loop", "tiles"), ("kit",))
            slot_id = _str(sm["id"], f"{sw}.id", _SLOT)
            if slot_id.rsplit(".", 1)[0] != sid:
                raise _fail(f"{sw}.id", f"{slot_id!r} is not a slot of {sid}")
            loop = _str(sm["loop"], f"{sw}.loop")
            if not _LOOP.fullmatch(loop) or loop not in loop_names:
                raise _fail(f"{sw}.loop", f"{loop!r} is not one of L1..L9")
            tiles = _strs(sm["tiles"], f"{sw}.tiles")
            for t in tiles:
                if t not in symbols:
                    raise _fail(f"{sw}.tiles", f"{t!r} is not a symbol of {sid}: tiles hold symbols, never options")
            if not 2 <= len(tiles) <= MAX_TILES:
                raise _fail(f"{sw}.tiles", f"a round shows 2-{MAX_TILES} clips")
            slots.append(Slot(slot_id, sid, loop, tiles, _str(sm["kit"], f"{sw}.kit") if "kit" in sm else None))
        if len({s.id for s in slots}) != len(slots):
            raise _fail(f"{w}.slots", "a slot id is used twice")
        slot_ids = [s.id for s in slots]
        sub = {}
        if "sub" in m:
            for head, value in _map(m["sub"], f"{w}.sub", (), ("keys", "vamp")).items():
                sub[head] = _str(value, f"{w}.sub.{head}")
            if sub.get("vamp") not in (None, "option") or (sub.get("vamp") and not all(o in vamps for o in options)):
                raise _fail(f"{w}.sub", "`vamp: option` needs vamp options")
        if fills is None and sub.get("vamp") != "option":
            raise _fail(w, "a stage that fills no band slot substitutes its options' vamp line (sub: {vamp: option})")
        tie = None
        if "tie" in m:
            t = _map(m["tie"], f"{w}.tie", ("symbols", "tiles"))
            tie_syms = _strs(t["symbols"], f"{w}.tie.symbols")
            if not set(symbols) < set(tie_syms):
                raise _fail(f"{w}.tie.symbols", "a tie adds symbols to the stage's own")
            if not isinstance(t["tiles"], list) or len(t["tiles"]) != len(slots):
                raise _fail(f"{w}.tie.tiles", "a tie gives new tiles for each of the stage's slots")
            for j, row in enumerate(t["tiles"]):
                if set(_strs(row, f"{w}.tie.tiles[{j}]")) != set(tie_syms):
                    raise _fail(f"{w}.tie.tiles[{j}]", "a tie round shows every symbol the tie names")
            tie = MappingProxyType({"symbols": list(tie_syms), "tiles": [list(r) for r in t["tiles"]]})
        default = m.get("default")
        if default is not None and default not in options:
            raise _fail(f"{w}.default", f"{default!r} is not one of the stage's options")
        band = {}
        for slot, opt in _map(m.get("band") or {}, f"{w}.band", (), BAND_SLOTS).items():
            if slot == fills:
                raise _fail(f"{w}.band.{slot}", "the stage's own slot takes its options")
            band[slot] = None if opt is None else _str(opt, f"{w}.band.{slot}")
        if not isinstance(m.get("sax_muted", False), bool):
            raise _fail(f"{w}.sax_muted", "expected true or false")
        groups: list[Group] = []
        if group_specs:
            covered = [x for spec in group_specs for x in spec[0]]
            if sorted(covered) != sorted(slot_ids):
                raise _fail(f"{w}.groups", f"the groups must split the slots {slot_ids}, got {covered}")
            for g_slots, g_opts, g_draw, gw in group_specs:
                moods = {vamps[o].mood for o in g_opts if o in vamps}
                for s in slots:
                    if s.id in g_slots and moods and {loops[s.loop].brief.mood} != moods:
                        raise _fail(gw, f"slot {s.id}'s loop is {loops[s.loop].brief.mood}, but the group draws "
                                        f"{sorted(moods)} vamps: a round has one brief and one mood (CD-15)")
                groups.append(Group(symbols, g_opts, g_draw["rule"], g_slots, pin=_central_pin(g_draw),
                                    where=MappingProxyType({"mood": moods.pop()} if len(moods) == 1 else {})))
        else:
            groups = _groups_of(draw, symbols, options, tie, candidates)
        out[sid] = Stage(sid, _int(m["day"], f"{w}.day", 1, 3), _str(m["category"], f"{w}.category"), symbols,
                         options, draw, tuple(slots), bus, str(m.get("what", "")), MappingProxyType(sub), ranks, tie,
                         fills, default, MappingProxyType(band), bool(m.get("sax_muted", False)), None, tuple(groups))
    for st in list(out.values()):
        if st.ranks is not None:
            of = out.get(st.ranks["of"])
            if of is None:
                raise _fail(f"stages.{st.id}.ranks.of", f"{st.ranks['of']!r} is not a stage")
            if of.fills != st.fills or of.fills not in SLOT_BUS:
                raise _fail(f"stages.{st.id}.ranks.of", f"{of.id} must fill the same instrument slot as {st.id}")
            out[st.id] = replace(st, groups=tuple(replace(g, leaders=st.fills) for g in st.groups))
        for slot in st.slots:
            if slot.kit is not None:
                kits = next((s for s in out.values() if s.day == st.day and s.fills == "kit"), None)
                if kits is None or slot.kit not in kits.symbols:
                    raise _fail(f"stages.{st.id}.{slot.id}.kit", f"{slot.kit!r} is not a kit symbol of day {st.day}")
    for st in out.values():
        for slots in [st.slots, *st.slots_by_count.values()]:
            for slot in slots:
                for sym in slot.tiles:
                    hits = [g for g in st.groups if sym in g.symbols and g.covers(slot.id)]
                    if len(hits) != 1:
                        raise _fail(f"stages.{st.id}.{slot.id}", f"{sym!r} is not a symbol of exactly one draw group "
                                                                 f"({len(hits)} match)")
    return out


def _central_pin(draw: Mapping[str, Any]) -> Mapping[str, tuple[str, ...]]:
    if draw["rule"] != "central":
        return MappingProxyType({})
    return MappingProxyType({o: tuple(draw["symbols"]) for o in draw["options"]})


def _groups_of(draw: Mapping[str, Any], symbols: tuple[str, ...], options: tuple[str, ...],
               tie: Mapping[str, Any] | None, candidates: Mapping[str, Candidate]) -> list[Group]:
    """A one-draw stage's groups: one, or one per register (`registers`)."""
    rule = draw["rule"]
    if rule == "registers":
        return [Group(tuple(syms), tuple(o for o in options if candidates[o].register == reg), "any",
                      where=MappingProxyType({"register": reg}))
                for reg, syms in draw["groups"].items()]
    if rule == "family_pairs":
        return [Group(symbols, options, rule, pairs=tuple(tuple(p) for p in draw["pairs"].values()), by="family")]
    if rule == "ranks":
        return [Group(tuple(tie["symbols"]) if tie else symbols, (), rule)]
    return [Group(symbols, options, rule, pin=_central_pin(draw))]


def _draw(node: Any, where: str, symbols: tuple[str, ...], options: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(node, dict) or node.get("rule") not in DRAW_RULES:
        raise _fail(where, f"expected a mapping with a rule in {list(DRAW_RULES)}")
    rule = node["rule"]
    extra = {"any": (), "ranks": (), "family_pairs": ("pairs",), "registers": ("groups",),
             "central": ("options", "symbols")}[rule]
    m = _map(node, where, ("rule",) + extra)
    if rule == "family_pairs":
        pairs = _map(m["pairs"], f"{where}.pairs", tuple(m["pairs"]) if isinstance(m["pairs"], dict) else ())
        flat = [s for p in pairs.values() for s in _strs(p, f"{where}.pairs")]
        if sorted(flat) != sorted(symbols) or any(len(p) != 2 for p in pairs.values()):
            raise _fail(f"{where}.pairs", "the pairs must split the symbols into twins")
    elif rule == "registers":
        groups = _map(m["groups"], f"{where}.groups", ("tenor", "alto"))
        flat = [s for g in groups.values() for s in _strs(g, f"{where}.groups")]
        if sorted(flat) != sorted(symbols):
            raise _fail(f"{where}.groups", "the register groups must split the symbols")
    elif rule == "central":
        if not set(_strs(m["options"], f"{where}.options")) <= set(options):
            raise _fail(f"{where}.options", "names an option the stage lacks")
        if not set(_strs(m["symbols"], f"{where}.symbols")) <= set(symbols):
            raise _fail(f"{where}.symbols", "names a symbol the stage lacks")
    return MappingProxyType({k: (dict(v) if isinstance(v, dict) else v) for k, v in m.items()})


def _days(node: Any, stages: Mapping[str, Stage], candidates: Mapping[str, Candidate],
          band_options: Mapping[str, tuple[str, ...]]) -> dict[int, Day]:
    if not isinstance(node, dict) or not node or not set(node) <= {1, 2, 3}:
        raise _fail("days", "expected a mapping of days among 1, 2 and 3")
    orphans = sorted(s.id for s in stages.values() if s.day not in node)
    if orphans:
        raise _fail("days", f"stages {orphans} belong to a day the plan does not list")
    out = {}
    for d, raw in node.items():
        w = f"days.{d}"
        m = _map(raw, w, ("date", "stages", "band"), ("leaders", "fixed"))
        names = _strs(m["stages"], f"{w}.stages")
        for s in names:
            if s not in stages or stages[s].day != d:
                raise _fail(f"{w}.stages", f"{s!r} is not a day-{d} stage")
        if {s.id for s in stages.values() if s.day == d} != set(names):
            raise _fail(f"{w}.stages", "must list every stage of the day")
        band = _map(m["band"], f"{w}.band", BUSES)
        for bus, inst in band.items():
            _str(inst, f"{w}.band.{bus}")
            if d == 1 and (inst not in candidates or candidates[inst].bus != bus):
                raise _fail(f"{w}.band.{bus}", f"{inst!r} is not a {bus} candidate")
        slots: dict[str, dict[str, str | None]] = {}
        for key in ("leaders", "fixed"):
            slots[key] = {}
            for slot, opt in _map(m.get(key) or {}, f"{w}.{key}", (), BAND_SLOTS).items():
                if opt is not None and opt not in band_options[slot]:
                    raise _fail(f"{w}.{key}.{slot}", f"{opt!r} is not one of the {SLOT_LIST[slot]} options")
                slots[key][slot] = opt
        both = set(slots["leaders"]) & set(slots["fixed"])
        if both:
            raise _fail(w, f"{sorted(both)} are both leaders and fixed")
        if d == 1 and slots["leaders"]:
            raise _fail(f"{w}.leaders", "day 1 plays the dev band (CD-7): no leaders")
        named = set(slots["leaders"]) | set(slots["fixed"]) | ({BUS_SLOT[b] for b in BUSES} if d == 1 else set())
        if named != set(BAND_SLOTS):
            raise _fail(w, f"the band leaves {sorted(set(BAND_SLOTS) - named)} unnamed")
        out[d] = Day(d, _str(m["date"], f"{w}.date"), names, MappingProxyType(dict(band)),
                     MappingProxyType(slots["leaders"]), MappingProxyType(slots["fixed"]))
    return out


def _batches(node: Any, stages: Mapping[str, Stage]) -> tuple[Batch, ...]:
    if not isinstance(node, list) or not node:
        raise _fail("batches", "expected a non-empty list")
    out = []
    for i, raw in enumerate(node):
        w = f"batches[{i}]"
        m = _map(raw, w, ("names", "date", "day", "rounds", "anchors", "category", "note", "est_min"), ("split",))
        names = _strs(m["names"], f"{w}.names")
        for n in names:
            if not BATCH_RE.fullmatch(n):
                raise _fail(f"{w}.names", f"batch name {n!r} must match {BATCH_RE.pattern}")
        day = _int(m["day"], f"{w}.day", 1, 3)
        rounds = {}
        for s, count in _map(m["rounds"], f"{w}.rounds", tuple(m["rounds"]) if isinstance(m["rounds"], dict)
                             else ()).items():
            if s not in stages:
                raise _fail(f"{w}.rounds", f"{s!r} is not a stage")
            if stages[s].day != day:
                raise _fail(f"{w}.rounds", f"stage {s} belongs to day {stages[s].day}, not day {day}")
            rounds[s] = _int(count, f"{w}.rounds.{s}", 1, 12)
        est = m["est_min"]
        if not isinstance(est, list) or len(est) != 2:
            raise _fail(f"{w}.est_min", "expected [plan, full]")
        out.append(Batch(names, _str(m["date"], f"{w}.date"), day, MappingProxyType(rounds),
                         _int(m["anchors"], f"{w}.anchors", 0, 3), _str(m["category"], f"{w}.category"),
                         _str(m["note"], f"{w}.note"), _num(est[0], f"{w}.est_min[0]"), _num(est[1], f"{w}.est_min[1]"),
                         _str(m["split"], f"{w}.split", _ID) if "split" in m else None))
    return tuple(out)


def _cuts(node: Any, stages: Mapping[str, Stage]) -> tuple[Cut, ...]:
    slot_ids = {s.id: st for st in stages.values() for s in st.slots}
    out = []
    for i, raw in enumerate(node or []):
        w = f"cuts[{i}]"
        m = _map(raw, w, ("id", "day", "order", "drop"), ("retile", "default_on", "saves_min"))
        day = _int(m["day"], f"{w}.day", 1, 3)
        drop = _strs(m["drop"], f"{w}.drop")
        for s in drop:
            if s not in slot_ids:
                raise _fail(f"{w}.drop", f"{s!r} is not a slot")
            if slot_ids[s].day != day:
                raise _fail(f"{w}.drop", f"{s} is a day-{slot_ids[s].day} slot, not day {day}")
        retile = dict(m.get("retile") or {})
        for s, tiles in retile.items():
            if s not in slot_ids or s in drop:
                raise _fail(f"{w}.retile.{s}", "needs a slot the cut keeps")
            own = next(x for x in slot_ids[s].slots if x.id == s).tiles
            if sorted(tiles) != sorted(own):
                raise _fail(f"{w}.retile.{s}", f"must hold the slot's own symbols {list(own)}")
        default_on = dict(m.get("default_on") or {})
        for s, pair in default_on.items():
            if (s not in stages or not isinstance(pair, list) or len(pair) != 2 or pair[0] not in stages[s].symbols
                    or pair[1] not in stages[s].options):
                raise _fail(f"{w}.default_on.{s}", "needs a stage, one of its symbols and one of its options")
        out.append(Cut(_str(m["id"], f"{w}.id"), day, _int(m["order"], f"{w}.order", 1, 9), drop,
                       MappingProxyType(retile), MappingProxyType(default_on),
                       tuple(float(x) for x in m.get("saves_min") or ())))
    if len({c.id for c in out}) != len(out):
        raise _fail("cuts", "a cut id is used twice")
    return tuple(out)


def _levels(node: Any, candidates: Mapping[str, Candidate]) -> Levels:
    m = _map(node, "levels", ("anchors", "sax_target", "zero_trim_source", "linear_drop_db", "linear_tol_db",
                              "median_tol_lu", "loop_tol_lu"))
    anchors = _map(m["anchors"], "levels.anchors", BUSES)
    for bus, cid in anchors.items():
        if cid not in candidates or candidates[cid].bus != bus:
            raise _fail(f"levels.anchors.{bus}", f"{cid!r} is not a {bus} candidate")
    if m["sax_target"] not in candidates or candidates[m["sax_target"]].bus != "sax":
        raise _fail("levels.sax_target", f"{m['sax_target']!r} is not a sax candidate")
    return Levels(MappingProxyType(dict(anchors)), m["sax_target"], _str(m["zero_trim_source"], "levels.zero_trim_source"),
                  *(_num(m[k], f"levels.{k}") for k in ("linear_drop_db", "linear_tol_db", "median_tol_lu",
                                                         "loop_tol_lu")))


# ------------------------------------------------------------------------------------------------ cuts
def apply_cuts(plan: Plan, names: Sequence[str]) -> Plan:
    """The plan with the named cuts of §2.7 applied (a new Plan; `plan` is unchanged): each cut's slots dropped (a
    stage left with none is dropped too), its kept slots retiled, and each `default_on` option pinned to its symbol
    (the default on the middle tile sum). Batches and days keep only the stages left, with their new round counts."""
    stages = dict(plan.stages)
    for name in names:
        cut = plan.cut(name)
        dropped = set(cut.drop)
        for sid, st in list(stages.items()):
            ids = {s.id for s in st.slots}
            if not ids & (dropped | set(cut.retile)) and sid not in cut.default_on:
                continue
            slots = tuple(replace(s, tiles=tuple(cut.retile.get(s.id, s.tiles))) for s in st.slots
                          if s.id not in dropped)
            if not slots:
                del stages[sid]
                continue
            kept = {s.id for s in slots}
            groups = tuple(replace(g, slots=tuple(x for x in g.slots if x in kept)) for g in st.groups)
            if sid in cut.default_on:
                sym, opt = cut.default_on[sid]
                groups = tuple(replace(g, pin=MappingProxyType({**g.pin, opt: (sym,)}))
                               if sym in g.symbols and opt in g.options else g for g in groups)
            stages[sid] = replace(st, slots=slots, groups=groups)
    days = {d: replace(day, stages=tuple(s for s in day.stages if s in stages)) for d, day in plan.days.items()}
    batches = tuple(replace(b, rounds=MappingProxyType({s: len(stages[s].slots) for s in b.rounds if s in stages}))
                    for b in plan.batches)
    return replace(plan, stages=MappingProxyType(stages), days=MappingProxyType(days), batches=batches,
                   applied_cuts=plan.applied_cuts + tuple(names))
