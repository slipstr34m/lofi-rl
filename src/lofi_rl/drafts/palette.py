"""The draft palette (docs/DRAFTS.md §3): the Day-2 picks as one palette, and the palette each draft renders with.

    python -m lofi_rl.drafts.palette build --levels-day1 <job 1 levels.json> --levels-day2 <job 2 levels.json>
        [--plan shootouts/plan.yaml] [--base shootouts/palettes/base.yaml] [--out palettes/draft_day2.yaml]
    python -m lofi_rl.drafts.palette check [--palette palettes/draft_day2.yaml]
        [--levels-day1 F --levels-day2 F [--plan P] [--base B]] [--verify]
    python -m lofi_rl.drafts.palette vocab [--palette palettes/draft_day2.yaml] [--json]

**What it holds** (gates/GATE_palette.md, Day 2, Shresth: "Yes, use these"): keys `rhodes` (gu_tine_ep), `wurli`
(gs_wurli_ep200) and `piano` (vcsl_upright_knight); kit `studio` (kit_virtuosity) with its bass bass_vsco2_pizz and kit
`jazz` (kit_gu_jazz) with bass_fp_finger_yr; the one sax `alto` (sax_mtg_alto) played plain (warmth off), expression
on, timing lag; swing 0.55; humanize off; no bed (Day 3 decides the rain); the chord list as it stands. Plain-word names
the drafts write (instrument.md §6); a loop's `sax:` line chooses nothing (DSL.md §4), so the drafts leave it out.

**How it is built** (`build`): the shootout palette generator's mechanism (SHOOTOUTS.md §7.8, CD-51), from the same
inputs: each instrument's block from shootouts/plan.yaml's candidate and the base palette (velocities, the alto
register, the GU kit's lanes, sources, chains, groove, sax_expr, loudness); the Virtuosity kit from the §7.4 importer's
printed blocks and its tar under $LOFI_ASSETS, a fetch-mode source like every non-GU source; the Day-2 settings as the
plan variants swing_55, humanize_off, sax_plain and timing_lag set them. Every source sha256, bank and program, and
each kit's lanes (canonical sha256), must equal what the levels files record for that instrument, so the palette plays
exactly what the shootouts measured.

**Trims and faders** (SHOOTOUTS.md §3 "Frozen values", CD-25, CD-26, CD-28): each instrument's t and median F from
job 2's levels.json (its re-measure of job 1's, every move 0.00 dB), except that a keys instrument's fader is the
mean of its rootless (K1) and spread (K2) median faders (CD-28: the drafts pick their voicing too). The sax's timing
lag keeps the instrument's trim and fader (CD-26). palette.py has no per-instrument fields until the freeze
(SHOOTOUTS.md §7.1), so:
- a kit's trim is in its lanes' `db` (CD-51), as in every shootout palette;
- every instrument's trim and fader, with where each number came from, is the `#| ` block at the top of the file: YAML
  in comments, which palette.py does not read and the file's sha256 pins with the rest;
- the palette as written plays the leaders' band (rhodes, studio): its chain-head trim slots and `bus_gain_db` are
  theirs, so it is a palette the renderer accepts as it stands;
- each draft renders with `render_doc(keys, kit)`: the palette narrowed to the draft's keys instrument and kit, with
  that band's trims in the chain-head `Gain` slots and its faders in `bus_gain_db`, exactly how a shootout clip's
  palette carries them.

**Validation** (`check`): the file parses (palette.py's schema, `parse_palette`), its `#| ` block covers exactly its
instruments, the palette as written equals render_doc(rhodes, studio) on every level, and each render palette parses;
with the levels files, a fresh build is byte-identical; with `--verify`, the shootout's own palette validation
(`lofi_rl.shootout.render.verify_palettes`: every source file hashed once against its sha256, every SoundFont preset
read from its file) and the SF2 coverage check (`Palette.verify_coverage`) on the palette and each render palette.

Sound claims: none.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from lofi_rl.dsl.vocab import EDIT_TARGETS, LANES, NUMERALS, QUALITIES, VOICINGS
from lofi_rl.palette import LAYERS, ROOT, Palette, PaletteError, load_palette, parse_palette

DRAFT_PALETTE = ROOT / "palettes" / "draft_day2.yaml"
PLAN = ROOT / "shootouts" / "plan.yaml"
BASE = ROOT / "shootouts" / "palettes" / "base.yaml"
LEVELS_SCHEMA = "lofi-rl/draft-levels/1"
INDEX_SCHEMA = "lofi-rl/draft-palettes/1"
BLOCK = "#| "
PALETTE_ID = "draft_day2"
TRIM_BUSES = ("sax", "keys", "bass")            # chains that start with the trim slot [Gain, {gain_db: t}] (CD-51)
BUS_ORDER = ("sax", "drums", "bass", "keys")    # mix.bus_gain_db's order in the base palette

# The Day-2 picks under the names the drafts write (gates/GATE_palette.md, Day 2). The first of each is the leader.
KEYS: dict[str, str] = {"rhodes": "gu_tine_ep", "wurli": "gs_wurli_ep200", "piano": "vcsl_upright_knight"}
KITS: dict[str, tuple[str, str]] = {"studio": ("kit_virtuosity", "bass_vsco2_pizz"),
                                    "jazz": ("kit_gu_jazz", "bass_fp_finger_yr")}
SAX: dict[str, str] = {"alto": "sax_mtg_alto"}
SETTINGS: tuple[str, ...] = ("swing_55", "humanize_off", "sax_plain", "timing_lag")   # plan.yaml variants, in order
LEADERS = ("rhodes", "studio")                  # the band the palette as written plays
JOBS = {"day1": 13635646, "day2": 13651610}     # the lofi-shootout jobs whose levels.json the numbers come from

# Seed data the writers may lean on, never rules (gates/GATE_palette.md, Day 2 "Vamps"; CD-45): each mood's vamps in
# Shresth's order, ties as one group; chords and span as shootouts/plan.yaml writes them.
VAMP_RANKS: dict[str, tuple[tuple[tuple[str, str, int], ...], ...]] = {
    "hopeful": ((("aruarian", "IIm9 V13", 2),), (("royal_half", "IVmaj7 IIIm7", 1),),
                (("lift", "Imaj9 IVmaj9", 2), ("two_of_us", "IVmaj7 III7", 2))),
    "melancholy": ((("plagal", "Im9 IVm9", 2),), (("rise", "bVImaj7 bVII13", 2),), (("backdoor", "IVm9 bVII13", 2),),
                   (("drift", "Im9 bVImaj7", 2),)),
    "wistful": ((("float", "Im9 bVIImaj7", 2),), (("i_iv", "Im9 IV13", 2), ("pulse", "Im9 IV9", 1)),
                (("step", "Im11 IIm7", 2),)),
}

_DUMP = {"sort_keys": False, "default_flow_style": None, "width": 120, "allow_unicode": True}


class DraftPaletteError(ValueError):
    pass


def yaml_text(doc: Any) -> str:
    """The pure-Python SafeDumper, so the bytes never depend on whether PyYAML was built with libyaml."""
    return yaml.dump(doc, Dumper=yaml.SafeDumper, **_DUMP)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    """The shootout tools' kit-map hash (lofi_rl.shootout.palettes.canonical_sha256): sorted keys, no spaces."""
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def mean2(a: float, b: float) -> float:
    """The exact mean of two decimals as written (CD-28): -6.65 and -6.22 give -6.435, never a binary neighbour."""
    return float((Decimal(repr(float(a))) + Decimal(repr(float(b)))) / 2)


# --------------------------------------------------------------------------------------------- the levels block
@dataclass(frozen=True)
class Level:
    option: str                 # the shootout candidate id (plan.yaml)
    bus: str                    # keys | drums | bass | sax
    trim_db: float
    fader_db: float


def levels_block(text: str) -> dict:
    """The `#| ` lines of a palette file, prefix stripped, as one YAML document. DraftPaletteError if there are none
    or they do not parse."""
    lines = [line[len(BLOCK):] if line.startswith(BLOCK) else "" for line in text.splitlines()
             if line.startswith(BLOCK) or line.rstrip() == BLOCK.rstrip()]
    if not lines:
        raise DraftPaletteError(f"no `{BLOCK.strip()}` levels block")
    try:
        doc = yaml.safe_load("\n".join(lines) + "\n")
    except yaml.YAMLError as err:
        raise DraftPaletteError(f"the levels block is not YAML: {err}") from err
    if not isinstance(doc, dict) or set(doc) != {"draft_levels"} or not isinstance(doc["draft_levels"], dict):
        raise DraftPaletteError("the levels block must be one mapping, `draft_levels:`")
    return doc["draft_levels"]


def _level(node: Any, where: str, bus: str) -> Level:
    if not isinstance(node, dict) or not {"option", "trim_db", "fader_db"} <= set(node):
        raise DraftPaletteError(f"levels {where}: needs option, trim_db and fader_db")
    vals = []
    for k in ("trim_db", "fader_db"):
        v = node[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not -60 <= v <= 60:
            raise DraftPaletteError(f"levels {where}.{k}: expected a number in -60..60 dB, got {v!r}")
        vals.append(float(v))
    if not isinstance(node["option"], str) or not node["option"]:
        raise DraftPaletteError(f"levels {where}.option: expected the shootout candidate id")
    return Level(node["option"], bus, *vals)


@dataclass(frozen=True)
class DraftPalette:
    path: Path
    text: str
    doc: dict                               # the YAML document as written
    palette: Palette
    levels: dict                            # the `#| ` block's draft_levels mapping
    keys: Mapping[str, Level]               # keys name -> its keys-bus level
    kits: Mapping[str, Level]               # kit name -> its drums-bus level (the trim is in its lanes' db)
    basses: Mapping[str, Level]             # kit name -> its bass's bass-bus level
    sax: Level

    @property
    def id(self) -> str:
        return self.palette.id

    @property
    def sha256(self) -> str:
        return self.palette.sha256

    def band(self, keys: str, kit: str) -> dict[str, Level]:
        """The level of each bus for a draft that names `keys` and `kit` (the palette's names)."""
        if keys not in self.keys:
            raise DraftPaletteError(f"keys {keys!r} is not in {self.id} ({sorted(self.keys)})")
        if kit not in self.kits:
            raise DraftPaletteError(f"kit {kit!r} is not in {self.id} ({sorted(self.kits)})")
        return {"sax": self.sax, "keys": self.keys[keys], "bass": self.basses[kit], "drums": self.kits[kit]}


def parse_draft(text: str, path: Path | str = DRAFT_PALETTE, *, verify: bool = False, root: Path = ROOT) -> DraftPalette:
    """A draft palette from its text: palette.py's schema, then the `#| ` block, which must give one level to every
    keys instrument, kit (and its bass) and sax the palette names, and nothing else."""
    path = Path(path)
    data = text.encode("utf-8")
    try:
        doc = yaml.safe_load(text)
        pal = parse_palette(copy.deepcopy(doc), str(path), sha256_bytes(data))
    except (yaml.YAMLError, PaletteError) as err:
        raise DraftPaletteError(f"{path}: {err}") from err
    if verify:
        problems = pal.verify_sources(root) or pal.verify_presets(root) or pal.verify_coverage(root)
        if problems:
            raise DraftPaletteError(f"{path}: " + "; ".join(problems))
    lv = levels_block(text)
    if lv.get("schema") != LEVELS_SCHEMA:
        raise DraftPaletteError(f"{path}: the levels block's schema must be {LEVELS_SCHEMA}")
    for slot, names in (("keys", pal.keys), ("kits", pal.kits), ("sax", pal.sax)):
        got = lv.get(slot)
        if not isinstance(got, dict) or set(got) != set(names):
            raise DraftPaletteError(f"{path}: levels.{slot} must name exactly the palette's {slot} "
                                    f"{sorted(names)}, got {sorted(got) if isinstance(got, dict) else got!r}")
    keys = {k: _level(v, f"keys.{k}", "keys") for k, v in lv["keys"].items()}
    kits, basses = {}, {}
    for k, v in lv["kits"].items():
        kits[k] = _level(v, f"kits.{k}", "drums")
        basses[k] = _level((v or {}).get("bass") if isinstance(v, dict) else None, f"kits.{k}.bass", "bass")
    sax = {k: _level(v, f"sax.{k}", "sax") for k, v in lv["sax"].items()}
    if len(sax) != 1 or pal.sax_lead not in sax:
        raise DraftPaletteError(f"{path}: one sax plays every loop (TASTE.md, Lead): the palette must hold exactly its "
                                f"sax_lead")
    return DraftPalette(path, text, doc, pal, lv, keys, kits, basses, sax[pal.sax_lead])


def load(path: Path | str = DRAFT_PALETTE, *, verify: bool = False, root: Path = ROOT) -> DraftPalette:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise DraftPaletteError(f"{p}: {err}") from err
    return parse_draft(text, p, verify=verify, root=root)


# --------------------------------------------------------------------------------------------- render palettes
def sources_of(doc: Mapping) -> set[str]:
    """Every source id a palette document's instruments play."""
    used = {s["src"] for s in doc["sax"].values()} | {k["src"] for k in doc["keys"].values()}
    for kit in doc["kits"].values():
        used.add(kit["bass"]["src"])
        for lane in kit["lanes"].values():
            used |= {s["src"] for layer in LAYERS for s in lane[layer]}
    if doc.get("bed"):
        used.add(doc["bed"]["src"])
    return used


def render_id(dp: DraftPalette, keys: str, kit: str) -> str:
    return f"{dp.id}_{keys}_{kit}"


def render_doc(dp: DraftPalette, keys: str, kit: str) -> dict:
    """The palette one draft renders with: `dp` narrowed to its keys instrument and kit (and that kit's bass), with the
    band's trims in the chain-head Gain slots and its faders in bus_gain_db (CD-51). The kit's trim is already in its
    lanes' db. Only the sources the band plays are kept, so a render hashes no file it does not use."""
    band = dp.band(keys, kit)
    doc = copy.deepcopy(dp.doc)
    doc["id"] = render_id(dp, keys, kit)
    doc["keys"] = {keys: doc["keys"][keys]}
    doc["kits"] = {kit: doc["kits"][kit]}
    used = sources_of(doc)
    doc["sources"] = {sid: s for sid, s in doc["sources"].items() if sid in used}
    for bus in TRIM_BUSES:
        chain = doc["mix"]["buses"][bus]
        if not chain or chain[0][0] != "Gain":
            raise DraftPaletteError(f"{dp.path}: mix.buses.{bus} must start with its trim slot [Gain, {{gain_db: t}}]")
        chain[0] = ["Gain", {"gain_db": band[bus].trim_db}]
    doc["mix"]["bus_gain_db"] = {bus: band[bus].fader_db for bus in BUS_ORDER}
    return doc


def render_text(dp: DraftPalette, keys: str, kit: str) -> str:
    doc = render_doc(dp, keys, kit)
    head = (f"# Draft render palette {doc['id']}: written by lofi_rl.drafts.palette from {dp.id} (sha256 "
            f"{dp.sha256[:12]}...) for keys {keys} and kit {kit}; do not edit.\n")
    text = head + yaml_text(doc)
    try:
        parse_palette(yaml.safe_load(text), doc["id"])
    except PaletteError as err:
        raise DraftPaletteError(f"render palette {doc['id']}: {err}") from err
    return text


def write_render_palettes(dp: DraftPalette, combos: Iterable[tuple[str, str]], out: Path | str) -> dict:
    """One render palette per (keys, kit) in `combos` into `out`, and out/index.json, which names each one's file,
    sha256, options and levels. Returns the index."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for keys, kit in sorted(set(combos)):
        text = render_text(dp, keys, kit)
        rid = render_id(dp, keys, kit)
        (out / f"{rid}.yaml").write_text(text, encoding="utf-8")
        band = dp.band(keys, kit)
        entries.append({"id": rid, "file": f"{rid}.yaml", "sha256": sha256_bytes(text.encode("utf-8")), "keys": keys,
                        "kit": kit, "options": {b: band[b].option for b in BUS_ORDER},
                        "trim_db": {b: band[b].trim_db for b in BUS_ORDER},
                        "fader_db": {b: band[b].fader_db for b in BUS_ORDER}})
    index = {"schema": INDEX_SCHEMA, "palette": {"path": _label(dp.path), "id": dp.id, "sha256": dp.sha256},
             "palettes": entries}
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return index


def verify_files(paths: list[Path], root: Path = ROOT) -> list[str]:
    """The shootout's validation of its palettes (lofi_rl.shootout.render.verify_palettes: each source file hashed once
    against its sha256, each SoundFont preset read from its file), then the SF2 coverage check load_palette(verify=True)
    adds (every note the compiler may send each voice has a zone). The problems, empty when all is well."""
    from lofi_rl.shootout.render import verify_palettes   # imported here: a shootout-side change breaks only this check
    problems = verify_palettes([Path(p) for p in paths], root)
    if problems:
        return problems
    for p in paths:
        try:
            problems += [f"{p}: {x}" for x in load_palette(p, verify=False, root=root).verify_coverage(root)]
        except (PaletteError, OSError) as err:
            problems.append(f"{p}: {err}")
    return problems


def _label(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------- build
def _json(path: Path) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    return json.loads(data.decode("utf-8")), sha256_bytes(data)


def _variant(levels: Mapping, option: str, what: str) -> Mapping:
    v = (levels.get("variants") or {}).get(option)
    if not isinstance(v, dict) or "t" not in v or "F" not in v:
        raise DraftPaletteError(f"the {what} levels file has no t and median F for {option}")
    return v


def _set_path(doc: dict, dotted: str, value: Any) -> None:
    node, keys = doc, dotted.split(".")
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            raise DraftPaletteError(f"variant path {dotted}: the base palette has no mapping {k}")
        node = node[k]
    if keys[-1] not in node:
        raise DraftPaletteError(f"variant path {dotted}: the base palette has no {keys[-1]}")
    node[keys[-1]] = copy.deepcopy(value)


def build(levels_day1: Path, levels_day2: Path, *, plan_path: Path = PLAN, base_path: Path = BASE,
          root: Path = ROOT) -> str:
    """The text of palettes/draft_day2.yaml from the plan, the base palette, the two levels files and the Virtuosity
    importer's outputs under the base palette's assets_root ($LOFI_ASSETS). DraftPaletteError when an input disagrees
    with what the levels files measured."""
    from lofi_rl.shootout.plan import load_plan        # the shootout's own reading of plan.yaml
    plan = load_plan(plan_path)
    base_bytes = Path(base_path).read_bytes()
    base_doc = yaml.safe_load(base_bytes.decode("utf-8"))
    base = parse_palette(copy.deepcopy(base_doc), str(base_path), sha256_bytes(base_bytes))
    day1, day1_sha = _json(levels_day1)
    day2, day2_sha = _json(levels_day2)
    if day1.get("stage") != "day1" or day2.get("stage") != "day2":
        raise DraftPaletteError("--levels-day1 must be job 1's levels.json (stage day1), --levels-day2 job 2's (day2)")
    if day2.get("base_sha256") != sha256_bytes(base_bytes):
        raise DraftPaletteError(f"{base_path} is not the base palette job 2 measured with (base_sha256 "
                                f"{str(day2.get('base_sha256'))[:12]}...)")
    if (day2.get("prior") or {}).get("sha256") != day1_sha:
        raise DraftPaletteError("--levels-day2 does not name --levels-day1 as its prior (job 2 compares with job 1)")
    if day1.get("S_star") != day2.get("S_star"):
        raise DraftPaletteError("S* moved between job 1 and job 2; the trims and faders need a re-measure (§3)")
    keys_vel = copy.deepcopy(base_doc["keys"][plan.voices["keys"]]["vel"])
    base_kit = base_doc["kits"][plan.voices["kit"]]
    bass_vel = copy.deepcopy(base_kit["bass"]["vel"])
    extra_sources: dict[str, dict] = {}
    record: dict[str, Any] = {"keys": {}, "kits": {}, "sax": {}}

    def numbers(option: str) -> tuple[float, float, Mapping]:
        """(t, median F) from job 2, and what both jobs measured: t and F each, and the instrument job 2 measured (its
        source's sha256, and its bank and program or its kit map's sha256), which `problems` checks offline."""
        v1, v2 = _variant(day1, option, "day-1"), _variant(day2, option, "day-2")
        what = {"source": v2.get("source"), "source_sha256": v2.get("source_sha256")}
        what.update({"kit_map_sha256": v2["kit_map_sha256"]} if v2.get("kit_map_sha256")
                    else {"bank": v2.get("bank"), "prog": v2.get("prog")})
        return float(v2["t"]), float(v2["F"]), {"day1": {"t": v1["t"], "F": v1["F"]},
                                                "day2": {"t": v2["t"], "F": v2["F"]}, "as": what}

    def sf_block(option: str, bus: str) -> dict:
        c = plan.candidates[option]
        if c.bus != bus or c.source is None:
            raise DraftPaletteError(f"{option} is not a SoundFont {bus} candidate of {plan_path}")
        v = _variant(day2, option, "day-2")
        sha = base.sources[c.source].sha256
        if (v.get("source"), v.get("source_sha256"), v.get("bank"), v.get("prog")) != (c.source, sha, c.bank, c.prog):
            raise DraftPaletteError(f"{option}: the plan and base palette give {c.source}@{sha[:12]} bank {c.bank} prog "
                                    f"{c.prog}, but job 2 measured {v.get('source')}@{str(v.get('source_sha256'))[:12]} "
                                    f"bank {v.get('bank')} prog {v.get('prog')}")
        return {"src": c.source, "bank": c.bank, "prog": c.prog}

    def kit_block(option: str, trim: float) -> dict:
        c = plan.candidates[option]
        v = _variant(day2, option, "day-2")
        if c.base:
            lanes, choke, src = copy.deepcopy(base_kit["lanes"]), copy.deepcopy(base_kit["choke"]), None
            shots = {s["src"] for lane in lanes.values() for layer in LAYERS for s in lane[layer]}
            if v.get("source_sha256") not in {base.sources[s].sha256 for s in shots}:
                raise DraftPaletteError(f"{option}: job 2 measured another source than the base kit's {sorted(shots)}")
        else:
            a = c.archive
            assets = base.assets_dir(root)
            tar, blocks = assets / a["path"], assets / a["blocks"]
            if not tar.is_file() or not blocks.is_file():
                raise DraftPaletteError(f"{option}: its §7.4 archive {a['path']} or blocks {a['blocks']} is missing under "
                                        f"{assets} (set LOFI_ASSETS)")
            sha = file_sha256(tar)
            printed = yaml.safe_load(blocks.read_text(encoding="utf-8"))
            lanes, choke = printed["lanes"], printed["choke"]
            srcs = {s["src"] for lane in lanes.values() for layer in LAYERS for s in lane[layer]}
            if len(srcs) != 1:
                raise DraftPaletteError(f"{option}: its one-shots name {sorted(srcs)}, not one archive")
            src = srcs.pop()
            extra_sources[src] = {"path": a["path"], "sha256": sha, "license": a["license"], "mode": "fetch",
                                  "url": None, "converted_from": a["kit_map"], "attribution": None}
        if canonical_sha256({"lanes": lanes, "choke": choke}) != v.get("kit_map_sha256"):
            raise DraftPaletteError(f"{option}: its lanes and choke are not the kit map job 2 measured")
        if src is not None and v.get("source_sha256") != sha:
            raise DraftPaletteError(f"{option}: {sha[:12]}... is not the archive job 2 measured")
        for lane in LANES:
            lanes[lane]["db"] = lanes[lane]["db"] + trim         # the kit's trim on its lanes (CD-51), as the shootout
        return {"choke": choke, "lanes": lanes}

    keys_doc = {}
    for name, option in KEYS.items():
        t, f_rootless, src = numbers(option)
        cfg = ((day2.get("configs") or {}).get(f"k2_{option}") or {}).get("buses", {}).get("keys", {})
        spread = (cfg.get("loops") or {}).get("F")
        if cfg.get("instrument") != option or spread is None:
            raise DraftPaletteError(f"{option}: job 2 has no K2 spread median fader (configs.k2_{option})")
        keys_doc[name] = {**sf_block(option, "keys"), "vel": copy.deepcopy(keys_vel)}
        record["keys"][name] = {"option": option, "trim_db": t, "fader_db": mean2(f_rootless, spread),
                                "rootless_F": f_rootless, "spread_F": float(spread), "rule": "CD-28: mean of K1 rootless "
                                "and K2 spread median F", "measured": src}
    kits_doc = {}
    for name, (kit, bass) in KITS.items():
        t, f, src = numbers(kit)
        bt, bf, bsrc = numbers(bass)
        kits_doc[name] = {"bass": {**sf_block(bass, "bass"), "vel": copy.deepcopy(bass_vel)}, **kit_block(kit, t)}
        record["kits"][name] = {"option": kit, "trim_db": t, "fader_db": f, "trim_in": "every lane's db (CD-51)",
                                "measured": src,
                                "bass": {"option": bass, "trim_db": bt, "fader_db": bf, "measured": bsrc}}
    sax_doc = {}
    for name, option in SAX.items():
        c = plan.candidates[option]
        if c.register != name:
            raise DraftPaletteError(f"{option} plays the {c.register} register, not {name}")
        t, f, src = numbers(option)
        sax_doc[name] = {**sf_block(option, "sax"), **copy.deepcopy(dict(plan.registers[name]))}
        record["sax"][name] = {"option": option, "trim_db": t, "fader_db": f, "rule": "median F; timing lag keeps the "
                               "instrument's trim and fader (CD-26)", "measured": src}

    doc = copy.deepcopy(base_doc)
    doc["id"] = PALETTE_ID
    used = {b["src"] for b in keys_doc.values()} | {k["bass"]["src"] for k in kits_doc.values()} \
        | {s["src"] for s in sax_doc.values()} | {"gu"}
    doc["sources"] = {sid: s for sid, s in base_doc["sources"].items() if sid in used}
    doc["sources"].update(extra_sources)
    doc["sax"], doc["sax_lead"] = sax_doc, next(iter(SAX))
    doc["keys"], doc["kits"], doc["aliases"] = keys_doc, kits_doc, {}
    for vid in SETTINGS:
        for dotted, value in plan.variants[vid].set.items():
            _set_path(doc, dotted, value)
    doc["bed"] = None
    lead_keys, lead_kit = LEADERS
    lead = {"sax": record["sax"][doc["sax_lead"]], "keys": record["keys"][lead_keys],
            "bass": record["kits"][lead_kit]["bass"], "drums": record["kits"][lead_kit]}
    for bus in TRIM_BUSES:
        if doc["mix"]["buses"][bus][0][0] != "Gain":
            raise DraftPaletteError(f"{base_path}: mix.buses.{bus} must start with its trim slot")
        doc["mix"]["buses"][bus][0] = ["Gain", {"gain_db": lead[bus]["trim_db"]}]
    doc["mix"]["bus_gain_db"] = {bus: lead[bus]["fader_db"] for bus in BUS_ORDER}

    levels = {"draft_levels": {
        "schema": LEVELS_SCHEMA, "S_star": day2["S_star"], "targets": dict(day2["targets"]),
        "from": {"day1": {"job": JOBS["day1"], "sha256": day1_sha}, "day2": {"job": JOBS["day2"], "sha256": day2_sha},
                 "base": {"path": _label(Path(base_path)), "sha256": sha256_bytes(base_bytes)},
                 "plan_settings": list(SETTINGS)},
        "as_written": {"keys": lead_keys, "kit": lead_kit},
        **record}}
    block = "".join(f"{BLOCK}{line}".rstrip() + "\n" for line in yaml_text(levels).splitlines())
    head = (
        "# Draft palette (docs/DRAFTS.md §3): the Day-2 picks (gates/GATE_palette.md, Shresth 2026-11-01: \"Yes, use\n"
        "# these\") under the names the drafts write. Written by python -m lofi_rl.drafts.palette build from\n"
        "# shootouts/plan.yaml, shootouts/palettes/base.yaml, the Virtuosity importer's outputs and the levels.json of\n"
        f"# lofi-shootout {JOBS['day1']} and {JOBS['day2']}; do not edit.\n"
        "# - keys: rhodes = gu_tine_ep, wurli = gs_wurli_ep200, piano = vcsl_upright_knight;\n"
        "# - kits: studio = kit_virtuosity with bass_vsco2_pizz, jazz = kit_gu_jazz with bass_fp_finger_yr;\n"
        "# - sax: alto = sax_mtg_alto, warmth off, expression on, timing lag; swing 0.55; humanize off; no bed (Day 3).\n"
        "# palette.py has no per-instrument trim or fader until the freeze (SHOOTOUTS.md §7.1), so each instrument's\n"
        "# trim and fader are in the `#| ` block below, which palette.py does not read and the file's sha256 pins. A\n"
        "# kit's trim is in its lanes' db (CD-51). As written the palette plays the leaders' band (rhodes, studio); each\n"
        "# draft renders with lofi_rl.drafts.palette.render_doc(keys, kit), the band's trims and faders in place.\n"
        "# Sound claims: none.\n")
    text = head + block + yaml_text(doc)
    parse_draft(text, DRAFT_PALETTE)                     # it must load as what it claims to be
    return text


# --------------------------------------------------------------------------------------------- checks
def problems(dp: DraftPalette) -> list[str]:
    """What is wrong with a draft palette without reading any other file. Every draft palette: the band its levels
    block says it plays as written (`as_written`) is the band its chain-head trims and faders play, and every render
    palette parses. palettes/draft_day2.yaml (id draft_day2) also: its names, options and settings are the Day-2 picks',
    the chord list is the grammar's, the band as written is the leaders', and each block is what job 2 measured
    (`picks_problems`). A test's stand-in palette under another id skips that second part."""
    out = []
    aw = dp.levels.get("as_written") or {}
    try:
        lead = render_doc(dp, aw.get("keys"), aw.get("kit"))
        mine = dp.doc["mix"]
        if lead["mix"]["bus_gain_db"] != mine["bus_gain_db"] or any(
                lead["mix"]["buses"][b][0] != mine["buses"][b][0] for b in TRIM_BUSES):
            out.append(f"the palette as written does not play its as_written band {aw} at that band's levels")
        for keys in dp.keys:
            for kit in dp.kits:
                render_text(dp, keys, kit)
    except DraftPaletteError as err:
        out.append(str(err))
    if dp.id == PALETTE_ID:
        out += picks_problems(dp)
    return out


def picks_problems(dp: DraftPalette) -> list[str]:
    """palettes/draft_day2.yaml against the Day-2 picks (gates/GATE_palette.md) and against what job 2 measured."""
    out = []
    pal = dp.palette
    if dict(dp.levels.get("as_written") or {}) != {"keys": LEADERS[0], "kit": LEADERS[1]}:
        out.append(f"levels.as_written must name the leaders' band {LEADERS}")
    if set(pal.keys) != set(KEYS) or set(pal.kits) != set(KITS) or set(pal.sax) != set(SAX):
        out.append(f"names: keys {sorted(pal.keys)}, kits {sorted(pal.kits)}, sax {sorted(pal.sax)}; the Day-2 picks are "
                   f"{sorted(KEYS)}, {sorted(KITS)}, {sorted(SAX)}")
    for name, option in KEYS.items():
        if name in dp.keys and dp.keys[name].option != option:
            out.append(f"keys {name} is {dp.keys[name].option}, not {option}")
    for name, (kit, bass) in KITS.items():
        if name in dp.kits and (dp.kits[name].option, dp.basses[name].option) != (kit, bass):
            out.append(f"kit {name} is {dp.kits[name].option} with {dp.basses[name].option}, not {kit} with {bass}")
    g, x = pal.groove, pal.sax_expr
    settings = {"swing": (str(g.swing), dict(g.swing_lanes)), "humanize": g.humanize, "sax_warm": pal.mix.sax_warm,
                "expression": x.enabled, "timing": x.timing, "bed": pal.bed, "aliases": dict(pal.aliases)}
    want = {"swing": ("11/20", {}), "humanize": "off", "sax_warm": False, "expression": True, "timing": "lag",
            "bed": None, "aliases": {}}
    out += [f"{k} is {settings[k]!r}, the Day-2 pick is {want[k]!r}" for k in want if settings[k] != want[k]]
    if tuple(pal.numerals) != NUMERALS or tuple(pal.qualities) != QUALITIES or tuple(pal.voicings) != VOICINGS:
        out.append("the chord list or the voicings are not the grammar's full lists (the list as it stands)")
    return out + _measured_problems(dp)


def untrimmed_lanes(lanes: Mapping, trim: float) -> dict:
    """A kit's lanes with its trim taken back off every lane's db (the importer's values, written to 2 decimals)."""
    out = copy.deepcopy(dict(lanes))
    for lane in LANES:
        out[lane]["db"] = round(out[lane]["db"] - trim, 9)
    return out


def _measured_problems(dp: DraftPalette) -> list[str]:
    """Each instrument block against what job 2 measured, as the `#| ` block records it (`measured.as`): a SoundFont
    voice's source sha256, bank and program; a kit's lanes, its trim taken off, by the shootout's kit-map hash, and its
    one-shots' source; and each level's numbers against the measured t and F (keys: CD-28's mean)."""
    out: list[str] = []
    srcs = dp.doc["sources"]

    def voice(where: str, block: Mapping, rec: Mapping) -> None:
        got = rec.get("measured", {}).get("as", {})
        mine = (block["src"], srcs[block["src"]]["sha256"], block["bank"], block["prog"])
        if mine != (got.get("source"), got.get("source_sha256"), got.get("bank"), got.get("prog")):
            out.append(f"{where}: plays {mine[0]}@{mine[1][:12]} bank {mine[2]} prog {mine[3]}, but job 2 measured "
                       f"{got.get('source')}@{str(got.get('source_sha256'))[:12]} bank {got.get('bank')} prog "
                       f"{got.get('prog')}")

    def numbers(where: str, rec: Mapping, fader: float) -> None:
        m = rec.get("measured", {}).get("day2", {})
        if rec.get("trim_db") != m.get("t") or fader != rec.get("fader_db"):
            out.append(f"{where}: trim {rec.get('trim_db')} and fader {rec.get('fader_db')} are not job 2's t "
                       f"{m.get('t')} and fader {fader}")

    lv = dp.levels
    for name, rec in (lv.get("keys") or {}).items():
        if name in dp.doc["keys"]:
            voice(f"keys.{name}", dp.doc["keys"][name], rec)
            if rec.get("rootless_F") != rec.get("measured", {}).get("day2", {}).get("F"):
                out.append(f"keys.{name}: rootless_F {rec.get('rootless_F')} is not job 2's median F")
            numbers(f"keys.{name}", rec, mean2(rec.get("rootless_F", 0.0), rec.get("spread_F", 0.0)))
    for name, rec in (lv.get("sax") or {}).items():
        if name in dp.doc["sax"]:
            voice(f"sax.{name}", dp.doc["sax"][name], rec)
            numbers(f"sax.{name}", rec, rec.get("measured", {}).get("day2", {}).get("F"))
    for name, rec in (lv.get("kits") or {}).items():
        if name not in dp.doc["kits"]:
            continue
        kit = dp.doc["kits"][name]
        voice(f"kits.{name}.bass", kit["bass"], rec.get("bass") or {})
        numbers(f"kits.{name}.bass", rec.get("bass") or {}, (rec.get("bass") or {}).get("measured", {}).get("day2", {})
                .get("F"))
        numbers(f"kits.{name}", rec, rec.get("measured", {}).get("day2", {}).get("F"))
        got = rec.get("measured", {}).get("as", {})
        lanes = untrimmed_lanes(kit["lanes"], float(rec.get("trim_db", 0.0)))
        if canonical_sha256({"lanes": lanes, "choke": kit["choke"]}) != got.get("kit_map_sha256"):
            out.append(f"kits.{name}: its lanes (trim {rec.get('trim_db')} dB taken off) and choke are not the kit map "
                       f"job 2 measured")
        shots = {s["src"] for lane in kit["lanes"].values() for layer in LAYERS for s in lane[layer]}
        if {srcs[s]["sha256"] for s in shots} != {got.get("source_sha256")}:
            out.append(f"kits.{name}: its one-shots' source is not the one job 2 measured")
    return out


def vocabulary(dp: DraftPalette) -> dict:
    """Everything a draft may write with this palette: names, chords (and the chords whose every tone fits each
    mood's mode), voicing styles, lanes, and the vamp ranks as seed data."""
    from lofi_rl.prompt import mood_chords, vocabulary as prompt_vocab
    v = prompt_vocab(dp.palette)
    return {"keys": {k: dp.keys[k].option for k in KEYS if k in dp.keys},
            "kits": {k: {"kit": dp.kits[k].option, "bass": dp.basses[k].option} for k in KITS if k in dp.kits},
            "sax": {k: dp.sax.option for k in dp.palette.sax}, "numerals": list(v["numerals"]),
            "qualities": list(v["qualities"]), "voicings": list(v["voicings"]), "drum_lanes": list(LANES),
            "edit_targets": list(EDIT_TARGETS),
            "chords_in_mode": {m: list(c) for m, c in mood_chords(dp.palette).items()},
            "vamp_ranks": {m: [[{"vamp": n, "chords": c, "span": s} for n, c, s in tie] for tie in ranks]
                           for m, ranks in VAMP_RANKS.items()}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.drafts.palette", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write palettes/draft_day2.yaml from the plan, the base and the levels files")
    c = sub.add_parser("check", help="validate the draft palette")
    v = sub.add_parser("vocab", help="print what a draft may write with the palette")
    for sp in (b, c):
        sp.add_argument("--levels-day1", type=Path, required=sp is b)
        sp.add_argument("--levels-day2", type=Path, required=sp is b)
        sp.add_argument("--plan", type=Path, default=PLAN)
        sp.add_argument("--base", type=Path, default=BASE)
    b.add_argument("--out", type=Path, default=DRAFT_PALETTE)
    for sp in (c, v):
        sp.add_argument("--palette", type=Path, default=DRAFT_PALETTE)
    c.add_argument("--verify", action="store_true", help="hash every source, read every preset and its coverage")
    v.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.cmd == "build":
            text = build(a.levels_day1, a.levels_day2, plan_path=a.plan, base_path=a.base)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(text, encoding="utf-8")
            print(f"wrote {a.out} (sha256 {sha256_bytes(text.encode('utf-8'))})")
            return 0
        dp = load(a.palette)
        if a.cmd == "vocab":
            voc = vocabulary(dp)
            if a.json:
                print(json.dumps(voc, indent=1))
            else:
                for k, val in voc.items():
                    print(f"{k}: {json.dumps(val)}")
            return 0
        bad = problems(dp)
        if a.levels_day1 or a.levels_day2:
            if not (a.levels_day1 and a.levels_day2):
                raise DraftPaletteError("--levels-day1 and --levels-day2 go together")
            fresh = build(a.levels_day1, a.levels_day2, plan_path=a.plan, base_path=a.base)
            if fresh != dp.text:
                bad.append(f"{a.palette} is not what build writes from these inputs today")
        if a.verify and not bad:
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                index = write_render_palettes(dp, [(k, kit) for k in dp.keys for kit in dp.kits], d)
                bad += verify_files([dp.path] + [Path(d) / e["file"] for e in index["palettes"]])
    except (DraftPaletteError, PaletteError, OSError, ValueError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    for x in bad:
        print(f"problem: {x}", file=sys.stderr)
    if bad:
        return 3
    combos = len(dp.keys) * len(dp.kits)
    print(f"ok: {dp.path} ({dp.id}, sha256 {dp.sha256}): keys {sorted(dp.keys)}, kits {sorted(dp.kits)}, sax "
          f"{sorted(dp.palette.sax)}; {combos} render palettes parse"
          + ("; sources, presets and coverage verified" if a.verify else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
