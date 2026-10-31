"""The shootout palette generator (docs/SHOOTOUTS.md §7.8; §3 "How the values are carried", CD-51).

    python -m lofi_rl.shootout.palettes --plan shootouts/plan.yaml --stage day1 --levels none|<levels.json>
        [--draw <draw.json>] [--base <base.yaml>] [--without <option>,...] --out <run>/summary/palettes/

Every palette is the committed base (`shootouts/palettes/base.yaml`) with one instrument per bus under the loops'
names (CD-1: keys `ep`, kit `dusty`, the sax named by its register and set as `sax_lead`), so a round's clips differ
only in the candidate's block, its trim and its fader (§2.1). How the values are carried (CD-51): a SoundFont
instrument's trim t is the trim slot `[Gain, {gain_db: t}]` heading its bus chain, a kit's trim is added to every
lane's `db`, and the fader F of each bus is `mix.bus_gain_db[bus]`. Every bus gets its own instrument's t and F, the
fixed band's included (§3 step 5).

- `--levels none` writes the measuring palettes for the level tool (§7.1) into <out>/measure/: run m1 holds the four
  level anchors, and each later run one not yet measured variant per bus (the anchor where a bus has none left), so
  each variant is measured once (§3 step 1: 8 runs on Day 1). Trims and faders are 0. measure/index.json says which
  variant each run measures on which bus.
- `--levels <levels.json>` writes one palette per (stage, option) of the day: <out>/<stage>/<option>.yaml, and for R2,
  whose rounds each play one kit, <out>/r2/<kit>/<bass>.yaml. <out>/index.json lists them with their instruments,
  trims, faders and sha256, and the day's round slots (loop, brief, tiles in symbols).
- `--draw <draw.json>` (optional) also writes <out>/clips.json, the render list: each slot's tiles resolved through
  the draw's symbol-to-option map to a palette. The draw is read as {"stage": "day1", "symbols": {stage: {symbol:
  option}}}; other keys are ignored. It must meet each stage's draw rule (plan.yaml).
- `--without` leaves options out (a kit whose §7.4 archive is not built yet, say) and records them; the band and the
  level anchors cannot be left out.

Kits from the §7.4 importer are read from $LOFI_ASSETS (the base's assets_root): the tar at the candidate's
`archive.path` and the importer's printed `lanes:` and `choke:` blocks saved at `archive.blocks`. The tar's sha256 is
checked against the plan's pin when the plan has one, and written into every palette as a fetch-mode source.

Every source file is hashed once and every SoundFont preset checked before anything is written; a generation that
fails writes nothing. Only Day 1 (job 1) is built: Day 2 needs the Day-1 leaders, Day 3 the rain file.
Exit codes: 0, 2 for a bad argument (a missing file, an unbuilt day), 3 when the inputs are refused.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import yaml

from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import LAYERS, ROOT, PaletteError, parse_palette, sf2_presets
from lofi_rl.shootout.plan import Plan, PlanError, load_plan

INDEX_SCHEMA = "lofi-rl/shootout-palettes/1"
MEASURE_SCHEMA = "lofi-rl/shootout-measure/1"
CLIPS_SCHEMA = "lofi-rl/shootout-clips/1"
LEVELS_SCHEMA = "lofi-rl/shootout-levels/1"
BUSES = ("sax", "keys", "bass", "drums")          # render/mix.py's summing order
TRIM_BUSES = ("sax", "keys", "bass")              # chains that start with the trim slot
STAGES = {"day1": 1, "day2": 2, "day3": 3}
BUILT = ("day1",)
DAY1_FLAGS = {"voicing": "rootless", "sax_warm": False, "sax_expr": True}   # §4 "Fixed in every round"


# libyaml when the PyYAML build has it: tests/test_shootout_palettes.py checks that it writes the same bytes as the
# pure-Python dumper, so a palette's sha256 does not depend on which one ran.
_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
DUMP_ARGS = {"sort_keys": False, "default_flow_style": None, "width": 120, "allow_unicode": True}


class GenerateError(ValueError):
    pass


def yaml_text(doc: Any, dumper: type = _DUMPER) -> str:
    return yaml.dump(doc, Dumper=dumper, **DUMP_ARGS)


def yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_LOADER)


@dataclass(frozen=True)
class Generated:
    index: dict
    files: tuple[Path, ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def label(path: Path, root: Path) -> str:
    """A path as the index records it: repo-relative when it lies in the repo, so a pulled index reads the same."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------- instruments
@dataclass(frozen=True)
class Instrument:
    """One candidate as a palette block. `block` is keys.ep, kits.dusty.bass or sax.<register> for a SoundFont
    instrument, or {lanes, choke} for a kit; `source` is the source its notes play (kits: the archive or gu)."""

    id: str
    bus: str
    block: Mapping[str, Any]
    source: str
    source_sha256: str
    register: str | None = None
    bank: int | None = None
    prog: int | None = None
    kit_map_sha256: str | None = None


class Inputs:
    """The plan, the base palette and every candidate of a day, resolved and checked once."""

    def __init__(self, plan: Plan, base_path: Path, day: int, without: Sequence[str], root: Path):
        self.plan, self.day, self.root = plan, day, root
        self.base_path = Path(base_path)
        data = self.base_path.read_bytes()
        self.base_sha256 = sha256_bytes(data)
        self.base_doc = yaml_load(data.decode("utf-8"))
        try:
            self.base = parse_palette(copy.deepcopy(self.base_doc), str(self.base_path), self.base_sha256)
        except PaletteError as err:
            raise GenerateError(f"base palette {self.base_path}: {err}") from err
        keys_name, kit_name = plan.voices["keys"], plan.voices["kit"]
        if set(self.base.keys) != {keys_name} or set(self.base.kits) != {kit_name} or len(self.base.sax) != 1:
            raise GenerateError(f"base palette {self.base_path}: needs exactly keys.{keys_name}, kits.{kit_name} "
                                f"and one sax (CD-1)")
        if self.base.aliases:
            raise GenerateError(f"base palette {self.base_path}: aliases must be empty (§7.8)")
        for bus in TRIM_BUSES:
            head = self.base_doc["mix"]["buses"][bus][0]
            if head[0] != "Gain" or set(head[1]) != {"gain_db"}:
                raise GenerateError(f"base palette {self.base_path}: mix.buses.{bus} must start with its trim slot "
                                    f"[Gain, {{gain_db: 0.0}}] (CD-51)")
        self.stages = plan.stages_of(day)
        options = {o for st in self.stages for o in st.options}
        protected = set(plan.days[day].band.values()) | set(plan.levels.anchors.values())
        for o in without:
            if o not in options:
                raise GenerateError(f"--without {o}: not an option of day {day}")
            if o in protected:
                raise GenerateError(f"--without {o}: the band and the level anchors cannot be left out")
        self.without = tuple(o for st in self.stages for o in st.options if o in set(without))
        self.options = {st.id: tuple(o for o in st.options if o not in self.without) for st in self.stages}
        self.extra_sources: dict[str, dict] = {}
        self.instruments = {o: self._instrument(o) for st in self.stages for o in self.options[st.id]}
        self._check_files()

    # ---------------------------------------------------------------------------------------------
    def _source_sha(self, sid: str) -> str:
        if sid in self.extra_sources:
            return self.extra_sources[sid]["sha256"]
        if sid not in self.base.sources:
            raise GenerateError(f"source {sid!r} is not in the base palette {self.base_path}")
        return self.base.sources[sid].sha256

    def _instrument(self, cid: str) -> Instrument:
        c = self.plan.candidates[cid]
        kit_name = self.plan.voices["kit"]
        if c.bus == "keys":
            block = {"src": c.source, "bank": c.bank, "prog": c.prog,
                     "vel": dict(self.base_doc["keys"][self.plan.voices["keys"]]["vel"])}
        elif c.bus == "bass":
            block = {"src": c.source, "bank": c.bank, "prog": c.prog,
                     "vel": dict(self.base_doc["kits"][kit_name]["bass"]["vel"])}
        elif c.bus == "sax":
            block = {"src": c.source, "bank": c.bank, "prog": c.prog, **copy.deepcopy(dict(self.plan.registers[c.register]))}
        else:
            return self._kit(cid)
        return Instrument(cid, c.bus, block, c.source, self._source_sha(c.source), c.register, c.bank, c.prog)

    def _kit(self, cid: str) -> Instrument:
        c = self.plan.candidates[cid]
        kit_name = self.plan.voices["kit"]
        base_kit = self.base_doc["kits"][kit_name]
        if c.base:
            block = {"lanes": copy.deepcopy(base_kit["lanes"]), "choke": copy.deepcopy(base_kit["choke"])}
            srcs = {s["src"] for lane in block["lanes"].values() for layer in LAYERS for s in lane[layer]}
            if len(srcs) != 1:
                raise GenerateError(f"kit {cid}: the base kit's one-shots name {sorted(srcs)}, not one source")
            src = srcs.pop()
            return Instrument(cid, "drums", block, src, self._source_sha(src), kit_map_sha256=canonical_sha256(block))
        a = c.archive
        assets = self.base.assets_dir(self.root)
        tar, blocks = assets / a["path"], assets / a["blocks"]
        for what, path in (("archive", tar), ("lanes and choke blocks", blocks)):
            if not path.is_file():
                raise GenerateError(f"kit {cid}: its §7.4 {what} {a['path'] if what == 'archive' else a['blocks']} is "
                                    f"missing under {assets} (python -m lofi_rl.convert.kit {a['kit_map']} writes it; "
                                    f"or leave the kit out with --without {cid})")
        sha = file_sha256(tar)
        if a["sha256"] is not None and sha != a["sha256"]:
            raise GenerateError(f"kit {cid}: {tar} does not match its pinned sha256 {a['sha256'][:12]}... "
                                f"(read {sha[:12]}...)")
        try:
            printed = yaml_load(blocks.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
            raise GenerateError(f"kit {cid}: {blocks}: {err}") from err
        if not isinstance(printed, dict) or set(printed) != {"lanes", "choke"}:
            raise GenerateError(f"kit {cid}: {blocks} must hold exactly the importer's `lanes:` and `choke:` blocks")
        if printed["choke"] != base_kit["choke"]:
            raise GenerateError(f"kit {cid}: choke {printed['choke']} is not dev_gu's {base_kit['choke']} (§4.2)")
        lanes = printed["lanes"]
        if not isinstance(lanes, dict) or set(lanes) != set(LANES):
            raise GenerateError(f"kit {cid}: {blocks} must give every lane {list(LANES)}")
        srcs = {s.get("src") for lane in lanes.values() for layer in LAYERS for s in (lane.get(layer) or [])}
        if len(srcs) != 1 or None in srcs:
            raise GenerateError(f"kit {cid}: every one-shot of {blocks} must name one source, the archive; got "
                                f"{sorted(map(str, srcs))}")
        src = srcs.pop()
        if src in self.base.sources or src in self.extra_sources:
            raise GenerateError(f"kit {cid}: source id {src!r} is already used")
        self.extra_sources[src] = {"path": a["path"], "sha256": sha, "license": a["license"], "mode": "fetch",
                                   "url": None, "converted_from": a["kit_map"], "attribution": None}
        block = {"lanes": lanes, "choke": printed["choke"]}
        return Instrument(cid, "drums", block, src, sha, kit_map_sha256=canonical_sha256(block))

    def _check_files(self) -> None:
        """Every source the palettes list matches its sha256, and every preset an instrument plays exists: each
        file is hashed once (§7.7)."""
        problems = []
        for sid, src in sorted(self.base.sources.items()):
            path = self.base.source_path(sid, self.root)
            if not path.is_file():
                problems.append(f"source {sid}: {path} is missing")
            elif file_sha256(path) != src.sha256:
                problems.append(f"source {sid}: {path} does not match its sha256")
        if problems:
            raise GenerateError("; ".join(problems))
        presets: dict[str, frozenset] = {}
        plays = [(i.id, i.source, i.bank, i.prog) for i in self.instruments.values() if i.bus != "drums"]
        for inst in self.instruments.values():
            if inst.bus == "drums":
                plays += [(inst.id, s["src"], s["bank"], s["prog"]) for lane in inst.block["lanes"].values()
                          for layer in LAYERS for s in lane[layer] if "file" not in s]
        for cid, src, bank, prog in plays:
            if src not in presets:
                try:
                    presets[src] = sf2_presets(self.base.source_path(src, self.root))
                except (OSError, PaletteError) as err:
                    raise GenerateError(f"{cid}: source {src}: {err}") from err
            if (bank, prog) not in presets[src]:
                raise GenerateError(f"{cid}: bank {bank} prog {prog} is not a preset of source {src}")

    # ------------------------------------------------------------------------------------- palettes
    def palette_doc(self, pid: str, band: Mapping[str, str], trims: Mapping[str, float],
                    faders: Mapping[str, float]) -> dict:
        """The base with `band` (bus -> instrument id) under the loops' names, each bus's trim and fader."""
        doc = copy.deepcopy(self.base_doc)
        doc["id"] = pid
        doc["sources"].update(copy.deepcopy(self.extra_sources))
        keys, kit_name = self.plan.voices["keys"], self.plan.voices["kit"]
        sax = self.instruments[band["sax"]]
        doc["keys"] = {keys: copy.deepcopy(dict(self.instruments[band["keys"]].block))}
        kit = self.instruments[band["drums"]]
        lanes = copy.deepcopy(kit.block["lanes"])
        for lane in LANES:
            lanes[lane]["db"] = lanes[lane]["db"] + trims["drums"]
        doc["kits"] = {kit_name: {"bass": copy.deepcopy(dict(self.instruments[band["bass"]].block)),
                                  "choke": copy.deepcopy(kit.block["choke"]), "lanes": lanes}}
        doc["sax"] = {sax.register: copy.deepcopy(dict(sax.block))}
        doc["sax_lead"] = sax.register
        for bus in TRIM_BUSES:
            doc["mix"]["buses"][bus][0] = ["Gain", {"gain_db": trims[bus]}]
        doc["mix"]["bus_gain_db"] = {b: faders[b] for b in self.base_doc["mix"]["bus_gain_db"]}
        return doc

    def dump(self, doc: dict, what: str) -> bytes:
        try:
            parse_palette(copy.deepcopy(doc), what)
        except PaletteError as err:
            raise GenerateError(f"{what}: {err}") from err
        head = (f"# Shootout palette {doc['id']}: written by lofi_rl.shootout.palettes (docs/SHOOTOUTS.md §7.8) from "
                f"the base palette, sha256 {self.base_sha256[:12]}...; do not edit.\n")
        return (head + yaml_text(doc)).encode("utf-8")

    def sources_index(self) -> dict:
        out = {sid: {"path": s.path, "sha256": s.sha256} for sid, s in sorted(self.base.sources.items())}
        out.update({sid: {"path": s["path"], "sha256": s["sha256"]} for sid, s in sorted(self.extra_sources.items())})
        return out


# --------------------------------------------------------------------------------------- measuring
def measuring_runs(inputs: Inputs) -> list[dict]:
    """§3 step 1: run m1 is the four level anchors; run r > 1 takes the (r-1)-th not-yet-measured variant of each
    bus, or the anchor where the bus has run out. `measure` lists the buses whose variant is new in that run."""
    anchors = inputs.plan.levels.anchors
    per_bus = {bus: [] for bus in BUSES}
    for st in inputs.stages:
        if st.bus in per_bus:
            per_bus[st.bus] += [o for o in inputs.options[st.id] if o != anchors[st.bus] and o not in per_bus[st.bus]]
    rows = [dict(anchors)]
    for col in zip_longest(*(per_bus[b] for b in BUSES)):
        rows.append({b: (v if v is not None else anchors[b]) for b, v in zip(BUSES, col)})
    seen: set[str] = set()
    runs = []
    for i, row in enumerate(rows, 1):
        measure = [b for b in BUSES if row[b] not in seen]
        seen |= {row[b] for b in measure}
        runs.append({"run": f"m{i}", "instruments": {b: row[b] for b in BUSES}, "measure": measure})
    return runs


def write_measuring(inputs: Inputs, out: Path, stage: str, plan_path: Path) -> Generated:
    zero = {b: 0.0 for b in BUSES}
    files, runs = {}, []
    for run in measuring_runs(inputs):
        pid = f"measure_{stage}_{run['run']}"
        data = inputs.dump(inputs.palette_doc(pid, run["instruments"], zero, zero), f"measure/{run['run']}.yaml")
        name = f"{run['run']}.yaml"
        files[Path("measure") / name] = data
        runs.append({"run": run["run"], "file": name, "id": pid, "sha256": sha256_bytes(data),
                     "instruments": run["instruments"], "measure": run["measure"]})
    first_run = {}
    for r in runs:
        for b in r["measure"]:
            first_run[r["instruments"][b]] = r["run"]
    variants = {}
    for vid, inst in sorted(inputs.instruments.items()):
        variants[vid] = {"bus": inst.bus, "source": inst.source, "source_sha256": inst.source_sha256,
                         "bank": inst.bank, "prog": inst.prog, "kit_map_sha256": inst.kit_map_sha256,
                         "flags": dict(DAY1_FLAGS), "run": first_run[vid]}
    index = {"schema": MEASURE_SCHEMA, "stage": stage, "plan": label(plan_path, inputs.root),
             "plan_sha256": inputs.plan.sha256, "base": label(inputs.base_path, inputs.root),
             "base_sha256": inputs.base_sha256, "without": list(inputs.without),
             "anchors": dict(inputs.plan.levels.anchors), "sources": inputs.sources_index(),
             "variants": variants, "runs": runs}
    files[Path("measure") / "index.json"] = (json.dumps(index, indent=1, sort_keys=False) + "\n").encode("utf-8")
    return Generated(index, write_all(out, files))


# --------------------------------------------------------------------------------------- stage palettes
def read_levels(path: Path, inputs: Inputs, stage: str) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise GenerateError(f"levels file {path}: {err}") from err
    if doc.get("schema") != LEVELS_SCHEMA or doc.get("stage") != stage:
        raise GenerateError(f"levels file {path}: expected {LEVELS_SCHEMA} for stage {stage}, got "
                            f"{doc.get('schema')!r} for {doc.get('stage')!r}")
    if doc.get("base_sha256") != inputs.base_sha256:
        raise GenerateError(f"levels file {path} was measured on base palette sha256 "
                            f"{str(doc.get('base_sha256'))[:12]}..., not {inputs.base_sha256[:12]}... (§3 drift)")
    variants = doc.get("variants") or {}
    for vid, inst in inputs.instruments.items():
        lv = variants.get(vid)
        if lv is None:
            raise GenerateError(f"levels file {path} has no trim and fader for {vid}")
        if lv.get("source_sha256") != inst.source_sha256 or lv.get("kit_map_sha256") != inst.kit_map_sha256:
            raise GenerateError(f"levels file {path}: {vid} was measured on another source or kit map; measure again")
        for k in ("t", "F"):
            if isinstance(lv.get(k), bool) or not isinstance(lv.get(k), (int, float)):
                raise GenerateError(f"levels file {path}: {vid}.{k} is not a number")
    return doc


def stage_entries(inputs: Inputs) -> list[dict]:
    """(stage, option, kit, instruments) of every palette of the day; R2 rounds each play one kit (§4.2)."""
    band = inputs.plan.days[inputs.day].band
    out = []
    for st in inputs.stages:
        if st.id == "R2":
            kits = next(s for s in inputs.stages if s.bus == "drums")
            for kit in inputs.options[kits.id]:
                for o in inputs.options[st.id]:
                    out.append({"stage": st.id, "option": o, "kit": kit,
                                "instruments": {**band, "drums": kit, st.bus: o}})
        else:
            out += [{"stage": st.id, "option": o, "kit": None, "instruments": {**band, st.bus: o}}
                    for o in inputs.options[st.id]]
    return out


def write_stage(inputs: Inputs, out: Path, stage: str, plan_path: Path, levels_path: Path,
                draw_path: Path | None) -> Generated:
    levels = read_levels(levels_path, inputs, stage)
    draw = read_draw(draw_path, inputs, stage) if draw_path is not None else None
    lv = levels["variants"]
    files, palettes = {}, []
    for e in stage_entries(inputs):
        inst = {b: e["instruments"][b] for b in BUSES}
        trims = {b: float(lv[inst[b]]["t"]) for b in BUSES}
        faders = {b: float(lv[inst[b]]["F"]) for b in BUSES}
        if e["kit"] is None:
            rel, pid = Path(e["stage"].lower()) / f"{e['option']}.yaml", f"{e['stage'].lower()}_{e['option']}"
        else:
            rel = Path(e["stage"].lower()) / e["kit"] / f"{e['option']}.yaml"
            pid = f"{e['stage'].lower()}_{e['kit']}_{e['option']}"
        data = inputs.dump(inputs.palette_doc(pid, inst, trims, faders), str(rel))
        files[rel] = data
        palettes.append({"stage": e["stage"], "option": e["option"], "kit": e["kit"], "file": rel.as_posix(),
                         "id": pid, "sha256": sha256_bytes(data), "instruments": inst, "trim_db": trims,
                         "fader_db": faders})
    briefs = inputs.plan.briefs()
    slots = []
    for st in inputs.stages:
        for s in st.slots:
            loop = inputs.plan.loop_names[s.loop]
            b = briefs[loop]
            slots.append({"stage": st.id, "id": s.id, "loop": s.loop, "loop_file": loop,
                          "brief": {"mood": b.mood, "energy": b.energy, "bpm": b.bpm, "tonic": b.tonic,
                                    "scene": b.scene},
                          "tiles": list(s.tiles), "kit": s.kit, "sub": dict(st.sub),
                          "text_sha256": sha256_bytes(inputs.plan.loop_text(s.loop).encode("utf-8"))})
    index = {"schema": INDEX_SCHEMA, "stage": stage, "plan": label(plan_path, inputs.root),
             "plan_sha256": inputs.plan.sha256, "base": label(inputs.base_path, inputs.root),
             "base_sha256": inputs.base_sha256,
             "levels": {"file": label(levels_path, inputs.root), "sha256": file_sha256(levels_path)}, "draw": None,
             "without": list(inputs.without), "band": dict(inputs.plan.days[inputs.day].band),
             "sources": inputs.sources_index(), "palettes": palettes, "slots": slots}
    if draw is not None:
        index["draw"] = {"file": label(draw_path, inputs.root), "sha256": file_sha256(draw_path)}
        files[Path("clips.json")] = (json.dumps(clips(inputs, draw, palettes, slots, index["draw"]), indent=1)
                                     + "\n").encode("utf-8")
    files[Path("index.json")] = (json.dumps(index, indent=1) + "\n").encode("utf-8")
    return Generated(index, write_all(out, files))


# --------------------------------------------------------------------------------------------- the draw
def read_draw(path: Path, inputs: Inputs, stage: str) -> dict[str, dict[str, str]]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise GenerateError(f"draw {path}: {err}") from err
    if not isinstance(doc, dict) or doc.get("stage") != stage or not isinstance(doc.get("symbols"), dict):
        raise GenerateError(f"draw {path}: expected {{'stage': {stage!r}, 'symbols': {{stage: {{symbol: option}}}}}}")
    maps = doc["symbols"]
    for st in inputs.stages:
        got = maps.get(st.id)
        if not isinstance(got, dict) or sorted(got) != sorted(st.symbols):
            raise GenerateError(f"draw {st.id}: must map exactly the symbols {list(st.symbols)}")
        if sorted(got.values()) != sorted(inputs.options[st.id]):
            raise GenerateError(f"draw {st.id}: must map onto the options {list(inputs.options[st.id])} one to one")
        check_rule(inputs.plan, st, got)
    return {st.id: dict(maps[st.id]) for st in inputs.stages}


def check_rule(plan: Plan, st, got: Mapping[str, str]) -> None:
    """The stage's draw rule (plan.yaml; §2.3 item 1, §4)."""
    rule = st.draw["rule"]
    cand = plan.candidates
    if rule == "family_pairs":
        families = []
        for pair, (a, b) in sorted(st.draw["pairs"].items()):
            fa, fb = cand[got[a]].family, cand[got[b]].family
            if fa != fb:
                raise GenerateError(f"draw {st.id}: pair {pair} ({a}, {b}) must hold the two members of one family, "
                                    f"got {got[a]} ({fa}) and {got[b]} ({fb})")
            families.append(fa)
        if len(set(families)) != len(families):
            raise GenerateError(f"draw {st.id}: two pairs hold the same family")
    elif rule == "registers":
        for reg, syms in st.draw["groups"].items():
            for sym in syms:
                if cand[got[sym]].register != reg:
                    raise GenerateError(f"draw {st.id}: {sym} is a {reg} symbol, but {got[sym]} is "
                                        f"{cand[got[sym]].register}")
    elif rule == "central":
        for o in st.draw["options"]:
            sym = next(s for s, v in got.items() if v == o)
            if sym not in st.draw["symbols"]:
                raise GenerateError(f"draw {st.id}: {o} must sit on {st.draw['symbols']}, not {sym} (CD-16)")


def clips(inputs: Inputs, draw, palettes, slots, draw_ref) -> dict:
    """The render list: every slot's tiles resolved through the draw to a palette (R2 takes its kit from R1's
    symbol)."""
    by_key = {(p["stage"], p["option"], p["kit"]): p for p in palettes}
    kit_stage = next(s.id for s in inputs.stages if s.bus == "drums")
    rounds = []
    for s in slots:
        kit_option = draw[kit_stage][s["kit"]] if s["kit"] else None
        tiles = []
        for pos, sym in enumerate(s["tiles"], 1):
            option = draw[s["stage"]][sym]
            p = by_key[(s["stage"], option, kit_option)]
            tiles.append({"tile": pos, "symbol": sym, "option": option, "palette": p["file"],
                          "palette_sha256": p["sha256"]})
        rounds.append({**{k: s[k] for k in ("stage", "id", "loop", "loop_file", "brief", "kit", "sub",
                                            "text_sha256")}, "kit_option": kit_option, "tiles": tiles})
    return {"schema": CLIPS_SCHEMA, "draw": draw_ref, "rounds": rounds}


# --------------------------------------------------------------------------------------------- writing
def write_all(out: Path, files: Mapping[Path, bytes]) -> tuple[Path, ...]:
    """Write every file only once all of them were built (a refused generation writes nothing)."""
    written = []
    for rel, data in files.items():
        path = Path(out) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written.append(path)
    return tuple(written)


def generate(plan_path: Path | str, out: Path | str, *, stage: str = "day1", levels: Path | str | None = None,
             draw: Path | str | None = None, base: Path | str | None = None, without: Sequence[str] = (),
             root: Path = ROOT) -> Generated:
    """Write the measuring palettes (levels=None) or the stage palettes of a day (§7.8). See the module docstring."""
    if stage not in STAGES:
        raise GenerateError(f"stage {stage!r} is not one of {list(STAGES)}")
    if stage not in BUILT:
        raise GenerateError(f"stage {stage}: only {', '.join(BUILT)} is built (job 1); {stage} needs "
                            f"{'the Day-1 leaders' if stage == 'day2' else 'the Day-2 picks and the rain file'}")
    try:
        plan = load_plan(plan_path, root=root)
    except PlanError as err:
        raise GenerateError(str(err)) from err
    inputs = Inputs(plan, Path(base) if base else root / plan.base, STAGES[stage], tuple(without), root)
    if levels is None:
        if draw is not None:
            raise GenerateError("--draw goes with the stage palettes (--levels <file>), not the measuring ones")
        return write_measuring(inputs, Path(out), stage, Path(plan_path))
    return write_stage(inputs, Path(out), stage, Path(plan_path), Path(levels), Path(draw) if draw else None)


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.shootout.palettes", description=__doc__.split("\n\n")[0])
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--levels", required=True, help="'none' for the measuring palettes, or the level tool's levels.json")
    p.add_argument("--draw", type=Path, default=None, help="the batch builder's draw.json: also write clips.json")
    p.add_argument("--base", type=Path, default=None, help="the base palette (default: the plan's)")
    p.add_argument("--without", default="", help="options to leave out, comma-separated (recorded in the index)")
    p.add_argument("--out", required=True, type=Path, help="<run>/summary/palettes/")
    a = p.parse_args(argv)
    if not a.plan.is_file():
        p.error(f"--plan {a.plan} is not a file")
    if a.stage not in BUILT:
        p.error(f"--stage {a.stage}: only {', '.join(BUILT)} is built so far (job 1)")
    levels = None if a.levels == "none" else Path(a.levels)
    for name, path in (("--levels", levels), ("--draw", a.draw), ("--base", a.base)):
        if path is not None and not path.is_file():
            p.error(f"{name} {path} is not a file")
    without = tuple(w for w in a.without.split(",") if w)
    try:
        got = generate(a.plan, a.out, stage=a.stage, levels=levels, draw=a.draw, base=a.base, without=without)
    except GenerateError as err:
        print(f"palettes: {err}", file=sys.stderr)
        return 3
    if levels is None:
        for r in got.index["runs"]:
            print(f"{r['run']} {r['sha256'][:12]} measures " + ", ".join(f"{b}={r['instruments'][b]}"
                                                                         for b in r["measure"]))
        print(f"wrote {len(got.index['runs'])} measuring palettes to {Path(a.out) / 'measure'}")
    else:
        for e in got.index["palettes"]:
            print(f"{e['file']} {e['sha256'][:12]} " + " ".join(f"{b}:t={e['trim_db'][b]:+.2f},F={e['fader_db'][b]:+.2f}"
                                                                for b in BUSES))
        print(f"wrote {len(got.index['palettes'])} palettes to {a.out}" + (", and clips.json" if a.draw else ""))
    if got.index["without"]:
        print("left out: " + ", ".join(got.index["without"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
