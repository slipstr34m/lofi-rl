"""shootouts/plan.yaml: the shootout plan in symbols (docs/SHOOTOUTS.md §4, §6; §7.8 "Plan").

`load_plan` validates the file strictly, as palette.py does a palette: unknown keys are refused, every reference
(a slot's loop and tiles, a stage's options, the band, a batch's stages, a cut's slots, the level anchors) must
resolve, and a tile may hold only a symbol of its stage, never an option (§2.3: the plan is public). Paths in the
plan are relative to the repo root.

Standard library and PyYAML only.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from lofi_rl.brief import Brief
from lofi_rl.palette import ROOT

BUSES = ("keys", "drums", "bass", "sax")
DRAW_RULES = ("any", "family_pairs", "registers", "central", "ranks")
_ID = re.compile(r"[a-z][a-z0-9_]*")
_STAGE = re.compile(r"[A-Z][A-Za-z0-9_]*")
_SLOT = re.compile(r"[A-Z][A-Za-z0-9_]*\.\d+")
_LOOP = re.compile(r"L[1-9]")
_SHA = re.compile(r"[0-9a-f]{64}")


class PlanError(ValueError):
    pass


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
        return f"{self.chords} span {self.span}"


@dataclass(frozen=True)
class Slot:
    id: str                                  # "K1.1"
    stage: str
    loop: str                                # L1..L9
    tiles: tuple[str, ...]                   # symbols, tile 1 first
    kit: str | None = None                   # R2: the R1 symbol whose kit plays in this round


@dataclass(frozen=True)
class Stage:
    id: str
    day: int
    category: str
    symbols: tuple[str, ...]
    options: tuple[str, ...]                 # () for a ranks stage: its options come from a tally (§8)
    draw: Mapping[str, Any] = field(hash=False)
    slots: tuple[Slot, ...] = ()
    bus: str | None = None
    what: str = ""
    sub: Mapping[str, str] = field(default_factory=dict, hash=False)
    ranks: Mapping[str, Any] | None = field(default=None, hash=False)
    tie: Mapping[str, Any] | None = field(default=None, hash=False)


@dataclass(frozen=True)
class Day:
    day: int
    date: str
    stages: tuple[str, ...]
    band: Mapping[str, str] = field(hash=False)     # bus -> candidate id (Day 1) or a tally phrase (Days 2-3)


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

    def briefs(self) -> dict[str, Brief]:
        """loop name -> its brief, from <loops>/briefs.yaml (§5)."""
        return read_briefs(self.loops_dir)

    def loop_file(self, name: str) -> Path:
        return self.loops_dir / f"{self.loop_names[name]}.ls"

    def loop_text(self, name: str) -> str:
        """The text of loop L1..L9 as committed."""
        return self.loop_file(name).read_text(encoding="utf-8")

    def stages_of(self, day: int) -> tuple[Stage, ...]:
        return tuple(self.stages[s] for s in self.days[day].stages)


def read_briefs(loops_dir: Path) -> dict[str, Brief]:
    doc = yaml.safe_load((Path(loops_dir) / "briefs.yaml").read_text(encoding="utf-8"))
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
    Substitutions). Each statement must appear exactly once, at the start of a line."""
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
def load_plan(path: Path | str, root: Path = ROOT) -> Plan:
    path = Path(path)
    data = path.read_bytes()
    try:
        doc = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as err:
        raise PlanError(f"{path}: not a UTF-8 YAML document ({err})") from err
    return parse_plan(doc, path, root=root, sha256=hashlib.sha256(data).hexdigest())


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


def parse_plan(doc: Any, path: Path | str = "", *, root: Path = ROOT, sha256: str = "") -> Plan:
    top = _map(doc, "plan", ("version", "base", "loops", "loop_names", "voices", "registers", "candidates",
                             "variants", "vamps", "stages", "days", "batches", "controls", "cuts", "levels"))
    if top["version"] != 1:
        raise _fail("version", f"expected 1, got {top['version']!r}")
    loops_dir = root / _str(top["loops"], "loops")
    names = _map(top["loop_names"], "loop_names", tuple(f"L{i}" for i in range(1, 10)))
    loop_names = {k: _str(v, f"loop_names.{k}", _ID) for k, v in names.items()}
    voices = _map(top["voices"], "voices", ("keys", "kit"))
    registers = {}
    for reg, r in _map(top["registers"], "registers", ("tenor", "alto")).items():
        m = _map(r, f"registers.{reg}", ("play", "comfort", "home"))
        registers[reg] = MappingProxyType({k: [_int(x, f"registers.{reg}.{k}", 0, 127) for x in m[k]] for k in m})
    candidates = _candidates(top["candidates"])
    variants = _variants(top["variants"])
    vamps = _vamps(top["vamps"])
    options = {**{k: "candidate" for k in candidates}, **{k: "variant" for k in variants},
               **{k: "vamp" for k in vamps}}
    if len(options) != len(candidates) + len(variants) + len(vamps):
        raise _fail("plan", "an option id is used by more than one of candidates, variants and vamps")
    stages = _stages(top["stages"], loop_names, candidates, variants, vamps)
    days = _days(top["days"], stages, candidates)
    batches = _batches(top["batches"], stages)
    cuts = _cuts(top["cuts"], stages)
    levels = _levels(top["levels"], candidates)
    controls = _map(top["controls"], "controls", ("repeats", "anchors"))
    for group, members in _map(controls["repeats"], "controls.repeats",
                               ("count", "groups", "day", "min_days_after", "tiles", "never"))["groups"].items():
        for s in members:
            if s not in stages:
                raise _fail(f"controls.repeats.groups.{group}", f"{s!r} is not a stage")
    _map(controls["anchors"], "controls.anchors", ("count", "per_day", "sour", "gate"))
    return Plan(path=Path(path), sha256=sha256, root=root, base=_str(top["base"], "base"), loops_dir=loops_dir,
                loop_names=MappingProxyType(loop_names), voices=MappingProxyType(dict(voices)),
                registers=MappingProxyType(registers), candidates=MappingProxyType(candidates),
                variants=MappingProxyType(variants), vamps=MappingProxyType(vamps),
                stages=MappingProxyType(stages), days=MappingProxyType(days), batches=batches,
                controls=MappingProxyType(controls), cuts=cuts, levels=levels)


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


def _stages(node: Any, loop_names: Mapping[str, str], candidates: Mapping[str, Candidate],
            variants: Mapping[str, Variant], vamps: Mapping[str, Vamp]) -> dict[str, Stage]:
    if not isinstance(node, dict) or not node:
        raise _fail("stages", "expected a non-empty mapping")
    out: dict[str, Stage] = {}
    for sid, raw in node.items():
        w = f"stages.{sid}"
        _str(sid, f"{w} id", _STAGE)
        m = _map(raw, w, ("day", "category", "symbols", "draw", "slots"),
                 ("bus", "what", "options", "ranks", "sub", "tie"))
        symbols = _strs(m["symbols"], f"{w}.symbols")
        bus = None
        if "bus" in m:
            bus = _str(m["bus"], f"{w}.bus")
            if bus not in BUSES:
                raise _fail(f"{w}.bus", f"{bus!r} is not one of {list(BUSES)}")
        if ("options" in m) == ("ranks" in m):
            raise _fail(w, "a stage names its options, or the ranks of an earlier stage, not both")
        options: tuple[str, ...] = ()
        ranks = None
        if "options" in m:
            options = _strs(m["options"], f"{w}.options")
            for o in options:
                if o in candidates:
                    if bus is None or candidates[o].bus != bus:
                        raise _fail(f"{w}.options", f"candidate {o!r} plays bus {candidates[o].bus}, not {bus}")
                elif o in variants:
                    if variants[o].stage != sid:
                        raise _fail(f"{w}.options", f"variant {o!r} belongs to stage {variants[o].stage}")
                elif o not in vamps:
                    raise _fail(f"{w}.options", f"{o!r} is not a candidate, variant or vamp")
            if len(options) != len(symbols):
                raise _fail(w, f"{len(symbols)} symbols for {len(options)} options")
        else:
            r = _map(m["ranks"], f"{w}.ranks", ("of", "top"))
            ranks = MappingProxyType({"of": _str(r["of"], f"{w}.ranks.of", _STAGE),
                                      "top": _int(r["top"], f"{w}.ranks.top", 1, 4)})
        draw = _draw(m["draw"], f"{w}.draw", symbols, options)
        slots = []
        if not isinstance(m["slots"], list) or not m["slots"]:
            raise _fail(f"{w}.slots", "expected a non-empty list")
        for i, s in enumerate(m["slots"]):
            sw = f"{w}.slots[{i}]"
            sm = _map(s, sw, ("id", "loop", "tiles"), ("kit",))
            slot_id = _str(sm["id"], f"{sw}.id", _SLOT)
            if slot_id.split(".")[0] != sid:
                raise _fail(f"{sw}.id", f"{slot_id!r} is not a slot of {sid}")
            loop = _str(sm["loop"], f"{sw}.loop")
            if not _LOOP.fullmatch(loop) or loop not in loop_names:
                raise _fail(f"{sw}.loop", f"{loop!r} is not one of L1..L9")
            tiles = _strs(sm["tiles"], f"{sw}.tiles")
            for t in tiles:
                if t not in symbols:
                    raise _fail(f"{sw}.tiles", f"{t!r} is not a symbol of {sid}: tiles hold symbols, never options")
            if not 2 <= len(tiles) <= 4:
                raise _fail(f"{sw}.tiles", "a round shows 2-4 clips")
            slots.append(Slot(slot_id, sid, loop, tiles, _str(sm["kit"], f"{sw}.kit") if "kit" in sm else None))
        if len({s.id for s in slots}) != len(slots):
            raise _fail(f"{w}.slots", "a slot id is used twice")
        sub = {}
        if "sub" in m:
            for head, value in _map(m["sub"], f"{w}.sub", (), ("keys", "vamp")).items():
                sub[head] = _str(value, f"{w}.sub.{head}")
            if sub.get("vamp") not in (None, "option") or (sub.get("vamp") and not all(o in vamps for o in options)):
                raise _fail(f"{w}.sub", "`vamp: option` needs vamp options")
        tie = None
        if "tie" in m:
            t = _map(m["tie"], f"{w}.tie", ("symbols", "tiles"))
            tie_syms = _strs(t["symbols"], f"{w}.tie.symbols")
            if not set(symbols) < set(tie_syms):
                raise _fail(f"{w}.tie.symbols", "a tie adds symbols to the stage's own")
            for j, row in enumerate(t["tiles"]):
                if not set(_strs(row, f"{w}.tie.tiles[{j}]")) <= set(tie_syms):
                    raise _fail(f"{w}.tie.tiles[{j}]", "holds a symbol the tie does not name")
            tie = MappingProxyType({"symbols": list(tie_syms), "tiles": [list(r) for r in t["tiles"]]})
        out[sid] = Stage(sid, _int(m["day"], f"{w}.day", 1, 3), _str(m["category"], f"{w}.category"), symbols,
                         options, draw, tuple(slots), bus, str(m.get("what", "")), MappingProxyType(sub), ranks, tie)
    for st in out.values():
        if st.ranks is not None and st.ranks["of"] not in out:
            raise _fail(f"stages.{st.id}.ranks.of", f"{st.ranks['of']!r} is not a stage")
        for slot in st.slots:
            if slot.kit is not None:
                kits = next((s for s in out.values() if s.day == st.day and s.bus == "drums"), None)
                if kits is None or slot.kit not in kits.symbols:
                    raise _fail(f"stages.{st.id}.{slot.id}.kit", f"{slot.kit!r} is not a kit symbol of day {st.day}")
    return out


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


def _days(node: Any, stages: Mapping[str, Stage], candidates: Mapping[str, Candidate]) -> dict[int, Day]:
    if not isinstance(node, dict) or sorted(node) != [1, 2, 3]:
        raise _fail("days", "expected days 1, 2 and 3")
    out = {}
    for d, raw in node.items():
        w = f"days.{d}"
        m = _map(raw, w, ("date", "stages", "band"))
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
        out[d] = Day(d, _str(m["date"], f"{w}.date"), names, MappingProxyType(dict(band)))
    return out


def _batches(node: Any, stages: Mapping[str, Stage]) -> tuple[Batch, ...]:
    if not isinstance(node, list) or not node:
        raise _fail("batches", "expected a non-empty list")
    out = []
    for i, raw in enumerate(node):
        w = f"batches[{i}]"
        m = _map(raw, w, ("names", "date", "day", "rounds", "anchors", "category", "note", "est_min"), ("split",))
        rounds = {}
        for s, count in _map(m["rounds"], f"{w}.rounds", tuple(m["rounds"]) if isinstance(m["rounds"], dict)
                             else ()).items():
            if s not in stages:
                raise _fail(f"{w}.rounds", f"{s!r} is not a stage")
            rounds[s] = _int(count, f"{w}.rounds.{s}", 1, 12)
        est = m["est_min"]
        if not isinstance(est, list) or len(est) != 2:
            raise _fail(f"{w}.est_min", "expected [plan, full]")
        out.append(Batch(_strs(m["names"], f"{w}.names"), _str(m["date"], f"{w}.date"), _int(m["day"], f"{w}.day", 1, 3),
                         MappingProxyType(rounds), _int(m["anchors"], f"{w}.anchors", 0, 3),
                         _str(m["category"], f"{w}.category"), _str(m["note"], f"{w}.note"),
                         _num(est[0], f"{w}.est_min[0]"), _num(est[1], f"{w}.est_min[1]"),
                         _str(m["split"], f"{w}.split") if "split" in m else None))
    return tuple(out)


def _cuts(node: Any, stages: Mapping[str, Stage]) -> tuple[Cut, ...]:
    slot_ids = {s.id: st for st in stages.values() for s in st.slots}
    out = []
    for i, raw in enumerate(node or []):
        w = f"cuts[{i}]"
        m = _map(raw, w, ("id", "day", "order", "drop"), ("retile", "default_on", "saves_min"))
        drop = _strs(m["drop"], f"{w}.drop")
        for s in drop:
            if s not in slot_ids:
                raise _fail(f"{w}.drop", f"{s!r} is not a slot")
        retile = dict(m.get("retile") or {})
        for s, tiles in retile.items():
            if s not in slot_ids or not set(tiles) <= set(slot_ids[s].symbols):
                raise _fail(f"{w}.retile.{s}", "needs a slot and its stage's symbols")
        default_on = dict(m.get("default_on") or {})
        for s, (sym, opt) in default_on.items():
            if s not in stages or sym not in stages[s].symbols or opt not in stages[s].options:
                raise _fail(f"{w}.default_on.{s}", "needs a stage, one of its symbols and one of its options")
        out.append(Cut(_str(m["id"], f"{w}.id"), _int(m["day"], f"{w}.day", 1, 3), _int(m["order"], f"{w}.order", 1, 9),
                       drop, MappingProxyType(retile), MappingProxyType(default_on),
                       tuple(float(x) for x in m.get("saves_min") or ())))
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
