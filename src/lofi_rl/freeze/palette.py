"""The frozen palette's generator and check (docs/FREEZE.md §3): `palettes/lofi_rain_v1.yaml` from its inputs.

    python -m lofi_rl.freeze build [--draft palettes/draft_day2.yaml] --levels-day3 <job 3's levels.json>
        [--plan shootouts/plan.yaml] [--rain shootouts/rain_excerpts.json] [--frozen null|YYYY-MM-DD]
        [--out palettes/lofi_rain_v1.yaml]
    python -m lofi_rl.freeze check [--palette palettes/lofi_rain_v1.yaml] [--gate gates/GATE_palette.md]
        [--levels-day1 F --levels-day2 F --levels-day3 F] [--verify] [--sealed]
    python -m lofi_rl.freeze vocab [--palette palettes/lofi_rain_v1.yaml] [--json]

**What it builds** (transform T, docs/FREEZE.md §3.3): the draft palette (the Day-2 picks under the drafts' plain-word
names, which keep3 played) with
1. `id: lofi_rain_v1` and `frozen:` null until Shresth's APPROVED line, then that date;
2. each instrument's `trim_db` and `fader_db` (SHOOTOUTS.md §7.1) from the draft's `#| ` record, and each kit's lanes
   with its trim taken back off (the importer's db);
3. no chain-head trim slots (the chains are dev_gu's) and `bus_gain_db` 0.0 on every bus;
4. the Day-3 rain: the excerpt of the W1 pick (excerpt_e2) at the level of the W2 pick (rain_d24), with the rain file
   as a fetch-mode source;
5. a `#| freeze_provenance:` block: every input's sha256, the draft's levels record verbatim, what Day 3 measured, the
   bed's record, the confirmation dates, the rules and any recorded change of Shresth's (lofi_rl.freeze.CHANGES).
Day 3's levels file must agree with Day 2's for every band instrument it measured, or the build stops. The build reads
no asset file, so it reruns anywhere; the YAML is the pure-Python SafeDumper's, so the bytes never depend on libyaml.

**The check** (docs/FREEZE.md §3.4): (a) it parses; (b) a fresh build from the committed inputs is byte-identical (Day
3's levels rebuilt from the provenance); (d) the names, options and settings are the picks'; (e) every number traces
to its record by its rule; (f) folded back into the draft's carriage, each (keys, kit) band is the draft's render
palette; (g) the regex; (h) prompt v2's 9 references; (i) `frozen:` is null, or with `--sealed` the APPROVED line's
date. `--gate` adds (c), the picks against gates/GATE_palette.md; `--levels-day1/2/3` the levels files themselves;
`--verify` load_palette(verify=True) on the fetched assets.

Sound claims: none.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from lofi_rl.drafts import palette as DP
from lofi_rl.drafts.palette import (BUS_ORDER, KEYS, KITS, SAX, TRIM_BUSES, DraftPalette, DraftPaletteError,
                                    canonical_sha256, file_sha256, mean2, render_doc, sha256_bytes, sources_of,
                                    untrimmed_lanes, yaml_text)
from lofi_rl.dsl.vocab import EDIT_TARGETS, LANES, NUMERALS, QUALITIES, VOICINGS
from lofi_rl.freeze import (BED, CHANGES, FROZEN, GATE, JOBS, LEVELS_DAY3, PALETTE_ID, PICKS, PROVENANCE_SCHEMA, RAIN,
                            REGEX_CHARS, REGEX_SHA256, V2_REFERENCES)
from lofi_rl.palette import LEVEL_FIELDS, ROOT, Palette, PaletteError, load_palette, parse_palette

DRAFT = DP.DRAFT_PALETTE
PLAN = DP.PLAN
DEV = ROOT / "palettes" / "dev_gu.yaml"          # whose chains the frozen palette plays (SHOOTOUTS.md:452-453)
BLOCK = DP.BLOCK
KEY = "freeze_provenance"
RAIN_SCHEMA = "lofi-rl/rain-excerpts/1"          # lofi_rl.shootout.palettes.RAIN_SCHEMA
RAIN_SOURCE = "rain"                             # lofi_rl.shootout.palettes.RAIN_SOURCE: the source id a bed plays
# The palette's top-level keys in palette.py's _TOP order (tests/test_freeze_palette.py checks it is _TOP's set).
ORDER = ("id", "frozen", "sr", "assets_root", "sources", "sax", "sax_lead", "keys", "kits", "aliases", "chords",
         "voicings", "groove", "sax_expr", "mix", "bed", "loudness")
KIT_KEYS = ("bass", "choke", "lanes")
# The Day-3 pick as gates/GATE_palette.md and its records give it: excerpt e2 (rain_excerpts.json, its start_s 44.3),
# the picker's crossfade 1.0 s, Δ 24 LU (rain_d24, "24 below (Recommended)").
PICKED_BED = {"excerpt": "e2", "src": RAIN_SOURCE, "start_s": 44.3, "xfade_s": 1.0, "delta_lu": 24.0}
# The Day-2 settings, the same assertions as lofi_rl.drafts.palette.picks_problems (swing 0.55 = 11/20 on every lane).
PICKED_SETTINGS = {"swing": ("11/20", {}), "humanize": "off", "sax_warm": False, "expression": True, "timing": "lag",
                   "aliases": {}, "sax_lead": "alto"}
# How each number of the palette follows from its record (the provenance block's `rules`; check (e) applies them).
# The texts are frozen with the file, so each says exactly what was compared: Day 1 re-measured all 8 instruments,
# Day 3 only the ones its `day3.measured` lists (5 of the 8; it never measured gs_wurli_ep200, vcsl_upright_knight or
# bass_fp_finger_yr).
RULES = {
    "trim_db": "t as Day 2 measured it (SHOOTOUTS.md §3, Frozen values). Day 1 measured the same t for every "
               "instrument; Day 3 measured the same t for those day3.measured lists, and did not measure the others",
    "fader_db": "the median F as Day 2 measured it (SHOOTOUTS.md §3). Day 1 measured the same F for every instrument; "
                "Day 3 measured the same F for those day3.measured lists, and did not measure the others",
    "keys_fader_db": "CD-28, the mean of the K1 rootless and K2 spread median F, exact (drafts.palette.mean2) and not "
                     "rounded to 0.01 dB, as keep3 played it (CD-29)",
    "sax": "timing lag keeps the trim and fader of the instrument (CD-26)",
    "bus_gain_db": "0.0 on every bus, each fader is the fader_db of its instrument (SHOOTOUTS.md §7.1)",
    "kit_lanes": "the importer db, the draft lanes with the kit trim taken off (drafts.palette.untrimmed_lanes)",
    "chains": "the draft chains without their chain-head trim slots, which are the dev_gu chains",
    "bed": "the excerpt of excerpt_e2 (W1) and only the delta_lu of rain_d24 (W2)",
}
CHANGE_RULES = ("keys_offset_db",)
_DATE = "%Y-%m-%d"

HEADER = (
    "# The frozen palette lofi_rain_v1 (docs/FREEZE.md): the shootout survivors under the names the model sees, with\n"
    "# their trims and faders, and the Day-3 rain (gates/GATE_palette.md: Shresth's confirmations of Days 1-3 and the\n"
    "# 10 drafts he kept in keep3).\n"
    "# After the freeze, palettes/lofi_rain_v1.yaml never changes (docs/DECISIONS.md D-017).\n"
    "# Written by python -m lofi_rl.freeze build from palettes/draft_day2.yaml, shootouts/plan.yaml,\n"
    f"# shootouts/rain_excerpts.json and the levels.json of lofi-shootout {JOBS['day3']}; do not edit.\n"
    "# `frozen:` is null until Shresth's line `APPROVED by Shresth Rana <yyyy-mm-dd>` in gates/GATE_palette.md, then\n"
    "# that date; only that line changes.\n"
    "# Each instrument carries its trim_db and fader_db; where every number came from is the `#| ` block below, which\n"
    "# palette.py does not read and the file's sha256 pins.\n"
    "# Sound claims: none.\n")


class FreezeError(ValueError):
    pass


def label(path: Path | str) -> str:
    """A path as the provenance records it: repo-relative when it lies in the repo."""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def dec_add(a: float, b: float) -> float:
    """The exact sum of two decimals as written (a recorded change, CD-28's style): never a binary neighbour."""
    return float(Decimal(repr(float(a))) + Decimal(repr(float(b))))


def _date(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise FreezeError(f"{where}: expected a date YYYY-MM-DD, got {value!r}")
    try:
        if _dt.datetime.strptime(value, _DATE).strftime(_DATE) != value:
            raise ValueError
    except ValueError as err:
        raise FreezeError(f"{where}: expected a date YYYY-MM-DD, got {value!r}") from err
    return value


# --------------------------------------------------------------------------------------------- the frozen file
@dataclass(frozen=True)
class FrozenPalette:
    path: Path
    text: str
    doc: dict                   # the YAML document as written
    palette: Palette
    provenance: dict            # the `#| ` block's freeze_provenance mapping

    @property
    def sha256(self) -> str:
        return self.palette.sha256


def provenance_block(text: str) -> dict:
    """The `#| ` lines of the file, prefix stripped, as one YAML document: `freeze_provenance:` with its schema."""
    lines = [line[len(BLOCK):] if line.startswith(BLOCK) else "" for line in text.splitlines()
             if line.startswith(BLOCK) or line.rstrip() == BLOCK.rstrip()]
    if not lines:
        raise FreezeError(f"no `{BLOCK.strip()}` provenance block")
    try:
        doc = yaml.safe_load("\n".join(lines) + "\n")
    except yaml.YAMLError as err:
        raise FreezeError(f"the provenance block is not YAML: {err}") from err
    if not isinstance(doc, dict) or set(doc) != {KEY} or not isinstance(doc[KEY], dict):
        raise FreezeError(f"the provenance block must be one mapping, `{KEY}:`")
    prov = doc[KEY]
    if prov.get("schema") != PROVENANCE_SCHEMA:
        raise FreezeError(f"the provenance block's schema must be {PROVENANCE_SCHEMA}, got {prov.get('schema')!r}")
    want = {"schema", "inputs", "draft_levels", "day3", "bed", "picks", "rules"}
    if set(prov) != want:
        raise FreezeError(f"the provenance block holds {sorted(prov)}, not {sorted(want)}")
    return prov


def parse_frozen(text: str, path: Path | str = FROZEN) -> FrozenPalette:
    """(a) A frozen palette from its text: palette.py's schema (parse_palette), then its provenance block."""
    path = Path(path)
    try:
        doc = yaml.safe_load(text)
        pal = parse_palette(copy.deepcopy(doc), str(path), sha256_bytes(text.encode("utf-8")))
    except (yaml.YAMLError, PaletteError) as err:
        raise FreezeError(f"{label(path)}: {err}") from err
    return FrozenPalette(path, text, doc, pal, provenance_block(text))


def load(path: Path | str = FROZEN) -> FrozenPalette:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise FreezeError(f"{p}: {err}") from err
    return parse_frozen(text, p)


# --------------------------------------------------------------------------------------------- inputs
def _json(path: Path | str, what: str) -> tuple[dict, str]:
    try:
        data = Path(path).read_bytes()
        doc = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as err:
        raise FreezeError(f"{what} {path}: {err}") from err
    if not isinstance(doc, dict):
        raise FreezeError(f"{what} {path}: expected a JSON object")
    return doc, sha256_bytes(data)


def _rain(doc: Mapping, path: str) -> None:
    if doc.get("schema") != RAIN_SCHEMA:
        raise FreezeError(f"{path}: expected a {RAIN_SCHEMA} record (python -m lofi_rl.convert.rain_excerpts)")
    src = doc.get("source") or {}
    for k in ("asset_path", "sha256", "pcm_sha256", "license"):
        if not src.get(k):
            raise FreezeError(f"{path}: source.{k} is missing")
    if not isinstance((doc.get("parameters") or {}).get("xfade_s"), (int, float)) or not doc.get("excerpts"):
        raise FreezeError(f"{path}: needs parameters.xfade_s and its excerpts")


def rain_source(record: Mapping) -> dict:
    """The rain file as a palette source, built as lofi_rl.shootout.palettes.read_rain builds it from the excerpt
    picker's record (mode fetch, under $LOFI_ASSETS), without hashing the file: the record pins its sha256."""
    src = record["source"]
    conv = src.get("converted_from") or {}
    return {"path": src["asset_path"], "sha256": src["sha256"], "license": src["license"], "mode": "fetch",
            "url": src.get("url"),
            "converted_from": (f"{conv['path']} (sha256 {str(conv.get('sha256'))[:12]}..., {conv.get('sr')} Hz) by "
                               "lofi_rl.convert.resample" if conv.get("path") else None),
            "attribution": src.get("attribution")}


def band_options() -> list[tuple[str, str]]:
    """(record path, shootout option) of every instrument, in the palette's order: keys, kits and their basses, sax."""
    out = [(f"keys.{k}", o) for k, o in KEYS.items()]
    for k, (kit, bass) in KITS.items():
        out += [(f"kits.{k}", kit), (f"kits.{k}.bass", bass)]
    return out + [(f"sax.{k}", o) for k, o in SAX.items()]


def _records(lv: Mapping) -> dict[str, Mapping]:
    """The draft's `#| ` record of each instrument, by its palette path (keys.rhodes, kits.studio.bass, sax.alto)."""
    out: dict[str, Mapping] = {f"keys.{k}": v for k, v in (lv.get("keys") or {}).items()}
    for k, v in (lv.get("kits") or {}).items():
        out[f"kits.{k}"], out[f"kits.{k}.bass"] = v, v.get("bass") or {}
    out.update({f"sax.{k}": v for k, v in (lv.get("sax") or {}).items()})
    return out


def _day3(lv: Mapping, day3: Mapping) -> dict[str, dict[str, float]]:
    """What Day 3 measured of the band, {option: {t, F}} in band order, after checking it against Day 2 (the draft's
    record): every band instrument Day 3 measured has Day 2's t, F and instrument (source sha256, bank and program or
    kit map), S* and the targets are Day 2's, nothing moved, and the band Day 3 played (the draft's leaders and its
    sax) is among them. FreezeError otherwise: any move stops the build."""
    if day3.get("stage") != "day3":
        raise FreezeError(f"--levels-day3 must be job 3's levels.json (stage day3), got stage {day3.get('stage')!r}")
    if day3.get("S_star") != lv["S_star"]:
        raise FreezeError(f"S* moved: Day 3 {day3.get('S_star')}, Day 2 {lv['S_star']}; the levels need a re-measure")
    if day3.get("targets") != lv["targets"]:
        raise FreezeError(f"the bus targets differ: Day 3 {day3.get('targets')}, Day 2 {lv['targets']}")
    if day3.get("moved") != []:
        raise FreezeError(f"Day 3 moved {day3.get('moved')!r}: any move stops the build")
    if (day3.get("prior") or {}).get("sha256") != lv["from"]["day1"]["sha256"]:
        raise FreezeError("--levels-day3 does not name job 1's levels.json as its prior (Day 3 compares with job 1)")
    if day3.get("base_sha256") != lv["from"]["base"]["sha256"]:
        raise FreezeError("Day 3 measured with another base palette than Day 2")
    recs = _records(lv)
    variants = day3.get("variants") or {}
    out: dict[str, dict[str, float]] = {}
    for where, option in band_options():
        v = variants.get(option)
        if v is None:
            continue
        rec = recs.get(where) or {}
        m = rec.get("measured") or {}
        d2, what = m.get("day2") or {}, m.get("as") or {}
        if rec.get("option") != option:
            raise FreezeError(f"{where}: the draft records {rec.get('option')!r}, not {option}")
        if (v.get("t"), v.get("F")) != (d2.get("t"), d2.get("F")):
            raise FreezeError(f"{option} moved: Day 3 t {v.get('t')} F {v.get('F')}, Day 2 t {d2.get('t')} F "
                              f"{d2.get('F')}; any move stops the build")
        diff = sorted(k for k in what if v.get(k) != what[k])
        if diff or not what:
            raise FreezeError(f"{option}: Day 3 measured another instrument than Day 2 ({diff or 'no record'})")
        out[option] = {"t": float(v["t"]), "F": float(v["F"])}
    lead_keys, lead_kit = (lv.get("as_written") or {}).get("keys"), (lv.get("as_written") or {}).get("kit")
    need = [KEYS.get(lead_keys), *KITS.get(lead_kit, (None, None)), *SAX.values()]
    missing = [o for o in need if o not in out]
    if missing:
        raise FreezeError(f"Day 3 did not measure {missing}, the band it played")
    return out


def _bed(plan: Any, rain: Mapping, bed_from: Mapping[str, str]) -> tuple[dict, dict]:
    """(the palette's bed, its provenance record): the excerpt of the W1 variant `excerpt_from`, played from its
    start_s with the picker's crossfade, at only the level (`bed.delta_lu`) of the W2 variant `delta_from`."""
    ex_id, dl_id = bed_from.get("excerpt_from"), bed_from.get("delta_from")
    ex_v, dl_v = plan.variants.get(ex_id), plan.variants.get(dl_id)
    if ex_v is None or ex_v.stage != "W1":
        raise FreezeError(f"the excerpt comes from the W1 pick: {ex_id!r} is not a W1 variant of the plan")
    if dl_v is None or dl_v.stage != "W2":
        raise FreezeError(f"the level comes from the W2 pick: {dl_id!r} is not a W2 variant of the plan")
    excerpt, delta = ex_v.set.get("bed.excerpt"), dl_v.set.get("bed.delta_lu")
    if not isinstance(excerpt, str):
        raise FreezeError(f"variant {ex_id} sets no bed.excerpt")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise FreezeError(f"variant {dl_id} sets no bed.delta_lu")
    ex = (rain.get("excerpts") or {}).get(excerpt)
    if not isinstance(ex, Mapping) or isinstance(ex.get("start_s"), bool) or \
            not isinstance(ex.get("start_s"), (int, float)) or not ex.get("pcm_sha256"):
        raise FreezeError(f"excerpt {excerpt!r} is not one of the picker's with a start_s and a pcm_sha256")
    bed = {"src": RAIN_SOURCE, "start_s": float(ex["start_s"]), "xfade_s": float(rain["parameters"]["xfade_s"]),
           "delta_lu": float(delta)}
    rec = {"excerpt": excerpt, "excerpt_from": ex_id, "delta_from": dl_id, "start_s": bed["start_s"],
           "xfade_s": bed["xfade_s"], "delta_lu": bed["delta_lu"], "excerpt_pcm_sha256": ex["pcm_sha256"],
           "file_sha256": rain["source"]["sha256"], "file_pcm_sha256": rain["source"]["pcm_sha256"]}
    return bed, rec


def changes_record(changes: Sequence[Any]) -> tuple[dict[str, float], dict[str, dict]]:
    """({keys name: dB offset}, {rule: provenance record}) from Shresth's recorded changes. Each needs his words (`by`
    Shresth, a `date`, his `words` verbatim); the one rule known is keys_offset_db with a dB for each keys name."""
    offsets: dict[str, float] = {}
    recs: dict[str, dict] = {}
    for i, c in enumerate(changes):
        where = f"CHANGES[{i}]"
        if not isinstance(c, Mapping):
            raise FreezeError(f"{where}: expected a mapping")
        missing = [k for k in ("rule", "by", "date", "words") if not c.get(k)]
        if missing:
            raise FreezeError(f"{where} has no {', '.join(missing)}: nothing changes without Shresth's words "
                              "(docs/FREEZE.md §3.5)")
        if c["by"] != "Shresth":
            raise FreezeError(f"{where}: by {c['by']!r}; only Shresth's words change an input")
        if not isinstance(c["words"], str) or not c["words"].strip():
            raise FreezeError(f"{where}: words must be his words, verbatim")
        _date(c["date"], f"{where}.date")
        if c["rule"] not in CHANGE_RULES:
            raise FreezeError(f"{where}: rule {c['rule']!r} is not one the build knows ({list(CHANGE_RULES)}); any "
                              "other change needs its own rule, check rule and tests in code first (docs/FREEZE.md "
                              "§3.5)")
        if c["rule"] in recs:
            raise FreezeError(f"{where}: a second {c['rule']} record; one per rule")
        extra = sorted(set(c) - {"rule", "by", "date", "words", "db"})
        db = c.get("db")
        if extra or not isinstance(db, Mapping) or set(db) != set(KEYS):
            raise FreezeError(f"{where}: keys_offset_db needs db with a number for each keys name {list(KEYS)}"
                              + (f" and nothing else ({extra})" if extra else ""))
        for k in KEYS:
            v = db[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or not -40 <= v <= 40:
                raise FreezeError(f"{where}.db.{k}: expected a number of dB in -40..40, got {v!r}")
            offsets[k] = float(v)
        recs[c["rule"]] = {"by": c["by"], "date": c["date"], "words": c["words"],
                           "db": {k: float(db[k]) for k in KEYS}}
    return offsets, recs


# --------------------------------------------------------------------------------------------- build
@dataclass(frozen=True)
class Inputs:
    draft: DraftPalette
    plan: Any                   # lofi_rl.shootout.plan.Plan
    rain: dict
    rain_sha256: str
    rain_label: str
    day3: dict
    day3_sha256: str


def read_inputs(levels_day3: Path | str | tuple[Mapping, str], *, draft: Path | str = DRAFT, plan: Path | str = PLAN,
                rain: Path | str = RAIN) -> Inputs:
    """The build's inputs, read and validated. `levels_day3` is job 3's levels.json, or (its document, its sha256)
    when it is rebuilt from a provenance block."""
    from lofi_rl.shootout.plan import PlanError, load_plan    # the shootout's own reading of plan.yaml
    try:
        dp = DP.load(draft)
    except DraftPaletteError as err:
        raise FreezeError(str(err)) from err
    try:
        pl = load_plan(Path(plan))
    except (PlanError, OSError, ValueError) as err:
        raise FreezeError(f"plan {plan}: {err}") from err
    rain_doc, rain_sha = _json(rain, "rain excerpts")
    _rain(rain_doc, label(rain))
    if isinstance(levels_day3, tuple):
        day3, day3_sha = dict(levels_day3[0]), str(levels_day3[1])
    else:
        day3, day3_sha = _json(levels_day3, "--levels-day3")
    if day3.get("plan_sha256") != pl.sha256:
        raise FreezeError(f"Day 3 ran with plan {str(day3.get('plan_sha256'))[:12]}..., not {label(plan)} "
                          f"({pl.sha256[:12]}...): the bed's variants must be the ones Day 3 played")
    return Inputs(dp, pl, rain_doc, rain_sha, label(rain), day3, day3_sha)


def _voice(block: Mapping, where: str, trim: float, fader: float) -> dict:
    if any(k in block for k in LEVEL_FIELDS):
        raise FreezeError(f"the draft's {where} already carries {[k for k in LEVEL_FIELDS if k in block]}")
    out = copy.deepcopy(dict(block))
    out["trim_db"], out["fader_db"] = float(trim), float(fader)
    return out


def build_text(inp: Inputs, *, frozen: str | None = None, changes: Sequence[Any] = CHANGES,
               bed_from: Mapping[str, str] = BED) -> str:
    """The text of palettes/lofi_rain_v1.yaml from read inputs (transform T, docs/FREEZE.md §3.3)."""
    if frozen is not None:
        _date(frozen, "frozen")
    dp, lv = inp.draft, inp.draft.levels
    bad = DP.problems(dp)
    if bad:
        raise FreezeError(f"{label(dp.path)} fails its own check: " + "; ".join(bad))
    if dp.id != DP.PALETTE_ID or tuple(dp.doc) != ORDER:
        raise FreezeError(f"{label(dp.path)} is not the draft palette {DP.PALETTE_ID} in palette.py's key order")
    measured = _day3(lv, inp.day3)
    bed, bed_rec = _bed(inp.plan, inp.rain, bed_from)
    offsets, change_recs = changes_record(changes)

    src = copy.deepcopy(dp.doc)
    doc: dict[str, Any] = {k: src[k] for k in ORDER}
    doc["id"], doc["frozen"] = PALETTE_ID, frozen
    if RAIN_SOURCE in doc["sources"]:
        raise FreezeError(f"the draft already has a {RAIN_SOURCE!r} source")
    doc["sources"][RAIN_SOURCE] = rain_source(inp.rain)
    for name, rec in lv["keys"].items():
        fader = mean2(rec["rootless_F"], rec["spread_F"])
        if fader != rec["fader_db"]:
            raise FreezeError(f"keys.{name}: the draft's fader {rec['fader_db']} is not CD-28's mean {fader}")
        doc["keys"][name] = _voice(src["keys"][name], f"keys.{name}", rec["trim_db"],
                                   dec_add(fader, offsets[name]) if name in offsets else fader)
    for name, rec in lv["sax"].items():
        doc["sax"][name] = _voice(src["sax"][name], f"sax.{name}", rec["trim_db"], rec["fader_db"])
    for name, rec in lv["kits"].items():
        kit = src["kits"][name]
        if tuple(kit) != KIT_KEYS:
            raise FreezeError(f"the draft's kits.{name} holds {list(kit)}, not {list(KIT_KEYS)}")
        doc["kits"][name] = {"trim_db": float(rec["trim_db"]), "fader_db": float(rec["fader_db"]),
                             "bass": _voice(kit["bass"], f"kits.{name}.bass", rec["bass"]["trim_db"],
                                            rec["bass"]["fader_db"]),
                             "choke": kit["choke"], "lanes": untrimmed_lanes(kit["lanes"], float(rec["trim_db"]))}
    mix = doc["mix"]
    for bus in TRIM_BUSES:
        chain = mix["buses"][bus]
        head = chain[0] if chain else None
        if not (isinstance(head, list) and len(head) == 2 and head[0] == "Gain" and isinstance(head[1], dict)
                and set(head[1]) == {"gain_db"}):
            raise FreezeError(f"mix.buses.{bus} must start with its trim slot [Gain, {{gain_db: t}}], got {head!r}")
        mix["buses"][bus] = chain[1:]
    for bus, chain in list(mix["buses"].items()) + [("master", mix["master"])]:
        if any(isinstance(p, list) and p and p[0] == "Gain" for p in chain):
            raise FreezeError(f"mix.buses.{bus}: a Gain is left in the chain; every trim is an instrument's trim_db")
    mix["bus_gain_db"] = {b: 0.0 for b in BUS_ORDER}
    doc["bed"] = bed

    prov = {KEY: {
        "schema": PROVENANCE_SCHEMA,
        "inputs": {"draft": {"path": label(dp.path), "sha256": dp.sha256},
                   "plan": {"path": label(inp.plan.path), "sha256": inp.plan.sha256},
                   "rain_excerpts": {"path": inp.rain_label, "sha256": inp.rain_sha256},
                   "levels_day3": {"path": LEVELS_DAY3, "sha256": inp.day3_sha256}},
        "draft_levels": copy.deepcopy(lv),
        "day3": {"job": JOBS["day3"], "levels_sha256": inp.day3_sha256, "S_star": inp.day3["S_star"],
                 "moved": list(inp.day3["moved"]), "measured": measured},
        "bed": bed_rec,
        "picks": {day: PICKS[day]["confirmation"]["date"] for day in ("day1", "day2", "day3")},
        "rules": {**RULES, **change_recs},
    }}
    block = "".join(f"{BLOCK}{line}".rstrip() + "\n" for line in yaml_text(prov).splitlines())
    text = HEADER + block + yaml_text(doc)
    parse_frozen(text, FROZEN)                       # it must load as what it claims to be
    return text


def build(levels_day3: Path | str | tuple[Mapping, str], *, draft: Path | str = DRAFT, plan: Path | str = PLAN,
          rain: Path | str = RAIN, frozen: str | None = None, changes: Sequence[Any] = CHANGES,
          bed_from: Mapping[str, str] = BED) -> str:
    """The text of palettes/lofi_rain_v1.yaml from the draft palette, the plan, the rain record and job 3's levels."""
    return build_text(read_inputs(levels_day3, draft=draft, plan=plan, rain=rain), frozen=frozen, changes=changes,
                      bed_from=bed_from)


def day3_from_provenance(prov: Mapping) -> tuple[dict, str]:
    """Job 3's levels file as far as the build reads it, rebuilt from a provenance block, with the sha256 the block
    records: so the rebuild runs on committed files alone (runs/ is never committed)."""
    lv, d3, inputs = prov["draft_levels"], prov["day3"], prov["inputs"]
    recs = _records(lv)
    variants = {}
    for where, option in band_options():
        if option in (d3.get("measured") or {}):
            variants[option] = {**dict((recs.get(where) or {}).get("measured", {}).get("as") or {}),
                                **d3["measured"][option]}
    doc = {"stage": "day3", "plan_sha256": inputs["plan"]["sha256"], "base_sha256": lv["from"]["base"]["sha256"],
           "prior": {"sha256": lv["from"]["day1"]["sha256"]}, "targets": lv["targets"], "S_star": d3["S_star"],
           "moved": d3["moved"], "variants": variants}
    return doc, d3["levels_sha256"]


def rebuild_text(fp: FrozenPalette, *, frozen: Any = "same", levels_day3: Path | str | None = None,
                 changes: Sequence[Any] = CHANGES, bed_from: Mapping[str, str] = BED, root: Path = ROOT) -> str:
    """What the build writes today for a frozen palette: from the committed inputs its provenance names (repo paths
    under `root`), this package's CHANGES and BED, and Day 3's levels rebuilt from the provenance unless
    `levels_day3` is given. `frozen`: "same" keeps the file's value; the seal passes Shresth's date."""
    inputs = fp.provenance["inputs"]
    day3 = levels_day3 if levels_day3 is not None else day3_from_provenance(fp.provenance)
    return build(day3, draft=root / inputs["draft"]["path"], plan=root / inputs["plan"]["path"],
                 rain=root / inputs["rain_excerpts"]["path"], frozen=fp.palette.frozen if frozen == "same" else frozen,
                 changes=changes, bed_from=bed_from)


def _changes_of(prov: Mapping) -> tuple[dict, ...]:
    """The changes a provenance block records, as CHANGES entries."""
    return tuple({"rule": r, **v} for r, v in prov["rules"].items() if r in CHANGE_RULES)


# --------------------------------------------------------------------------------------------- fold
def fold(doc: Mapping, keys: str, kit: str) -> dict:
    """The frozen document folded back into the draft's carriage (CD-51) for one band: narrowed to `keys` and `kit`,
    each trim back in its chain-head Gain slot (a kit's on every lane's db, the renderer's sum), each fader in
    bus_gain_db (0.0 + F, the renderer's sum), the fields and the bed gone, and only the sources the band plays."""
    d = copy.deepcopy(dict(doc))
    k = d["kits"][kit]
    band = {"sax": d["sax"][d["sax_lead"]], "keys": d["keys"][keys], "bass": k["bass"], "drums": k}
    out = {key: d[key] for key in ORDER}
    out["keys"] = {keys: {f: v for f, v in d["keys"][keys].items() if f not in LEVEL_FIELDS}}
    out["sax"] = {n: {f: v for f, v in s.items() if f not in LEVEL_FIELDS} for n, s in d["sax"].items()}
    lanes = copy.deepcopy(k["lanes"])
    for lane in lanes.values():
        lane["db"] = lane["db"] + k["trim_db"]
    out["kits"] = {kit: {"bass": {f: v for f, v in k["bass"].items() if f not in LEVEL_FIELDS}, "choke": k["choke"],
                         "lanes": lanes}}
    for bus in TRIM_BUSES:
        out["mix"]["buses"][bus] = [["Gain", {"gain_db": band[bus]["trim_db"]}]] + out["mix"]["buses"][bus]
    out["mix"]["bus_gain_db"] = {b: out["mix"]["bus_gain_db"][b] + band[b]["fader_db"] for b in BUS_ORDER}
    out["bed"] = None
    used = sources_of(out)
    out["sources"] = {sid: s for sid, s in out["sources"].items() if sid in used}
    return out


def fold_problems(fp: FrozenPalette, dp: DraftPalette) -> list[str]:
    """(f) For every (keys, kit): the fold is the draft's render palette (drafts.palette.render_doc) apart from `id`
    and `frozen` (which no render reads: the seal changes only that line), and with a recorded keys offset the keys
    bus_gain_db differs by exactly that offset (the exact decimal sum)."""
    out = []
    offsets, _ = changes_record(_changes_of(fp.provenance))
    for keys in fp.doc["keys"]:
        for kit in fp.doc["kits"]:
            try:
                want = render_doc(dp, keys, kit)
            except DraftPaletteError as err:
                out.append(f"fold {keys}/{kit}: {err}")
                continue
            if keys in offsets:
                want["mix"]["bus_gain_db"]["keys"] = dec_add(want["mix"]["bus_gain_db"]["keys"], offsets[keys])
            got = fold(fp.doc, keys, kit)
            got["id"], got["frozen"] = want["id"], want["frozen"]
            if json.dumps(got, sort_keys=True) != json.dumps(want, sort_keys=True):
                where = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
                out.append(f"fold {keys}/{kit} is not render_doc({dp.id}, {keys}, {kit}): {where} differ")
    return out


# --------------------------------------------------------------------------------------------- checks
def picks_problems(fp: FrozenPalette) -> list[str]:
    """(d) The names, options and settings are the picks' (lofi_rl.drafts.palette KEYS, KITS, SAX, as gates/
    GATE_palette.md gives them), with the same assertions as drafts.palette.picks_problems, and the Day-3 bed."""
    out = []
    pal, lv = fp.palette, fp.provenance["draft_levels"]
    if pal.id != PALETTE_ID:
        out.append(f"id is {pal.id!r}, not {PALETTE_ID}")
    if list(fp.doc["keys"]) != list(KEYS) or list(fp.doc["kits"]) != list(KITS) or list(fp.doc["sax"]) != list(SAX):
        out.append(f"names: keys {list(fp.doc['keys'])}, kits {list(fp.doc['kits'])}, sax {list(fp.doc['sax'])}; the "
                   f"picks are {list(KEYS)}, {list(KITS)}, {list(SAX)}")
    for name, option in KEYS.items():
        if (lv.get("keys") or {}).get(name, {}).get("option") != option:
            out.append(f"keys {name} is not {option}")
    for name, (kit, bass) in KITS.items():
        rec = (lv.get("kits") or {}).get(name) or {}
        if (rec.get("option"), (rec.get("bass") or {}).get("option")) != (kit, bass):
            out.append(f"kit {name} is not {kit} with {bass}")
    for name, option in SAX.items():
        if (lv.get("sax") or {}).get(name, {}).get("option") != option:
            out.append(f"sax {name} is not {option}")
    g, x = pal.groove, pal.sax_expr
    got = {"swing": (str(g.swing), dict(g.swing_lanes)), "humanize": g.humanize, "sax_warm": pal.mix.sax_warm,
           "expression": x.enabled, "timing": x.timing, "aliases": dict(pal.aliases), "sax_lead": pal.sax_lead}
    out += [f"{k} is {got[k]!r}, the pick is {v!r}" for k, v in PICKED_SETTINGS.items() if got[k] != v]
    want_bed = {k: v for k, v in PICKED_BED.items() if k != "excerpt"}
    if fp.doc.get("bed") != want_bed or fp.provenance["bed"].get("excerpt") != PICKED_BED["excerpt"]:
        out.append(f"bed is {fp.doc.get('bed')} (excerpt {fp.provenance['bed'].get('excerpt')!r}); the Day-3 pick is "
                   f"excerpt {PICKED_BED['excerpt']} {want_bed}")
    if tuple(pal.numerals) != NUMERALS or tuple(pal.qualities) != QUALITIES or tuple(pal.voicings) != VOICINGS:
        out.append("the chord list or the voicings are not the grammar's full lists (the list as it stands)")
    modes = {sid: s.mode for sid, s in pal.sources.items()}
    if modes.pop("gu", None) != "commit" or set(modes.values()) != {"fetch"}:
        out.append("GU is the one committed source and every other is fetch (D-012)")
    return out


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def numbers_problems(fp: FrozenPalette, dp: DraftPalette, dev: Mapping, root: Path = ROOT) -> list[str]:
    """(e) Every number traced: each field equals its record by its rule (t; the median F; CD-28's mean2; a recorded
    change added as the exact decimal sum), Day 3's measures equal the record; the chains are dev_gu's; the rest is the
    draft's; the bed is the record's excerpt at rain_d24's level; the sources are the draft's and the rain."""
    out: list[str] = []
    doc, prov = fp.doc, fp.provenance
    lv, d3 = prov["draft_levels"], prov["day3"]
    offsets, _ = changes_record(_changes_of(prov))
    if lv != dp.levels:
        out.append(f"draft_levels is not {label(dp.path)}'s `#| ` record")
    if d3.get("S_star") != lv.get("S_star") or d3.get("moved") != [] or d3.get("job") != JOBS["day3"]:
        out.append(f"day3: S* {d3.get('S_star')}, moved {d3.get('moved')}, job {d3.get('job')}; want Day 2's S* "
                   f"{lv.get('S_star')}, nothing moved, job {JOBS['day3']}")
    recs = _records(lv)
    blocks: dict[str, Mapping] = {f"keys.{k}": v for k, v in doc["keys"].items()}
    for k, v in doc["kits"].items():
        blocks[f"kits.{k}"], blocks[f"kits.{k}.bass"] = v, v["bass"]
    blocks.update({f"sax.{k}": v for k, v in doc["sax"].items()})
    if set(blocks) != {w for w, _ in band_options()} or set(recs) != set(blocks):
        out.append(f"instruments {sorted(blocks)} are not the record's {sorted(recs)}")
    for where, option in band_options():
        rec, block = recs.get(where) or {}, blocks.get(where) or {}
        m = rec.get("measured") or {}
        t, f = (m.get("day2") or {}).get("t"), (m.get("day2") or {}).get("F")
        if (m.get("day1") or {}) != {"t": t, "F": f}:
            out.append(f"{where}: Day 1 measured {m.get('day1')}, Day 2 t {t} F {f}")
        if option in (d3.get("measured") or {}) and d3["measured"][option] != {"t": t, "F": f}:
            out.append(f"{where}: Day 3 measured {d3['measured'][option]}, Day 2 t {t} F {f}")
        if where.startswith("keys."):
            if rec.get("rootless_F") != f:
                out.append(f"{where}: rootless_F {rec.get('rootless_F')} is not Day 2's median F {f}")
            want_f = mean2(rec.get("rootless_F", 0.0), rec.get("spread_F", 0.0))
            name = where.split(".", 1)[1]
            want_f = dec_add(want_f, offsets[name]) if name in offsets else want_f
        else:
            want_f = f
        if (rec.get("trim_db"), block.get("trim_db")) != (t, t) or block.get("fader_db") != want_f:
            rule = prov["rules"].get("keys_fader_db" if where.startswith("keys.") else "fader_db")
            out.append(f"{where}: trim {block.get('trim_db')} and fader {block.get('fader_db')} are not t {t} and "
                       f"fader {want_f} ({rule})")
        if where.startswith("kits.") and not where.endswith(".bass"):
            name = where.split(".", 1)[1]
            want_lanes = untrimmed_lanes(dp.doc["kits"][name]["lanes"], float(rec.get("trim_db", 0.0)))
            if _canon(block.get("lanes")) != _canon(want_lanes):
                out.append(f"{where}: its lanes are not the draft's with the trim {rec.get('trim_db')} dB taken off")
            if canonical_sha256({"lanes": block.get("lanes"), "choke": block.get("choke")}) != \
                    (m.get("as") or {}).get("kit_map_sha256"):
                out.append(f"{where}: its lanes and choke are not the kit map job 2 measured")
    mix, dmix, vmix = doc["mix"], dp.doc["mix"], dev["mix"]
    for bus in list(vmix["buses"]) + ["master"]:
        mine, theirs = (mix["master"], vmix["master"]) if bus == "master" else (mix["buses"].get(bus),
                                                                                  vmix["buses"][bus])
        if _canon(mine) != _canon(theirs):
            out.append(f"mix.{'master' if bus == 'master' else 'buses.' + bus} is not dev_gu's chain")
    if set(mix["buses"]) != set(vmix["buses"]):
        out.append(f"mix.buses names {sorted(mix['buses'])}, dev_gu {sorted(vmix['buses'])}")
    if _canon(mix["bus_gain_db"]) != _canon({b: 0.0 for b in BUS_ORDER}):
        out.append(f"mix.bus_gain_db is {mix['bus_gain_db']}: every bus is 0.0, each fader is its instrument's")
    for k in ("bus_target_lu", "sax_warm"):
        if _canon(mix[k]) != _canon(dmix[k]):
            out.append(f"mix.{k} is not the draft's")
    for k in ("sr", "assets_root", "sax_lead", "aliases", "chords", "voicings", "groove", "sax_expr", "loudness"):
        if _canon(doc[k]) != _canon(dp.doc[k]):
            out.append(f"{k} is not the draft's")
    for name, block in doc["keys"].items():
        if _canon({f: v for f, v in block.items() if f not in LEVEL_FIELDS}) != _canon(dp.doc["keys"].get(name)):
            out.append(f"keys.{name} is not the draft's voice")
    for name, block in doc["sax"].items():
        if _canon({f: v for f, v in block.items() if f not in LEVEL_FIELDS}) != _canon(dp.doc["sax"].get(name)):
            out.append(f"sax.{name} is not the draft's voice")
    for name, kit in doc["kits"].items():
        dk = dp.doc["kits"].get(name) or {}
        if _canon({f: v for f, v in kit["bass"].items() if f not in LEVEL_FIELDS}) != _canon(dk.get("bass")) or \
                _canon(kit["choke"]) != _canon(dk.get("choke")):
            out.append(f"kits.{name}: its bass or choke is not the draft's")
    rain, _ = _json(root / prov["inputs"]["rain_excerpts"]["path"], "rain excerpts")
    if _canon(doc["sources"]) != _canon({**dp.doc["sources"], RAIN_SOURCE: rain_source(rain)}):
        out.append("sources are not the draft's and the rain record's")
    if sources_of(doc) != set(doc["sources"]):
        out.append(f"sources {sorted(set(doc['sources']) - sources_of(doc))} play nothing")
    from lofi_rl.shootout.plan import load_plan
    plan = load_plan(root / prov["inputs"]["plan"]["path"])
    try:
        bed, bed_rec = _bed(plan, rain, {"excerpt_from": prov["bed"].get("excerpt_from"),
                                         "delta_from": prov["bed"].get("delta_from")})
        if doc["bed"] != bed or prov["bed"] != bed_rec:
            out.append(f"bed {doc['bed']} is not the record's excerpt {bed_rec['excerpt']} at the level of "
                       f"{bed_rec['delta_from']} ({bed})")
    except FreezeError as err:
        out.append(f"bed: {err}")
    if prov["bed"].get("excerpt_from") != BED["excerpt_from"] or prov["bed"].get("delta_from") != BED["delta_from"]:
        out.append(f"the bed comes from {prov['bed'].get('excerpt_from')} and {prov['bed'].get('delta_from')}, not "
                   f"{BED['excerpt_from']} (excerpt) and {BED['delta_from']} (level only)")
    if prov["picks"] != {day: PICKS[day]["confirmation"]["date"] for day in ("day1", "day2", "day3")}:
        out.append(f"picks {prov['picks']} are not the confirmation dates")
    if {k: v for k, v in prov["rules"].items() if k not in CHANGE_RULES} != RULES:
        out.append("rules are not the build's")
    return out


def inputs_problems(fp: FrozenPalette, root: Path = ROOT) -> list[str]:
    """Each committed input the provenance names still has the sha256 it records."""
    out = []
    for key in ("draft", "plan", "rain_excerpts"):
        rec = fp.provenance["inputs"].get(key) or {}
        path = root / str(rec.get("path"))
        try:
            got = file_sha256(path)
        except OSError as err:
            out.append(f"input {key}: {err}")
            continue
        if got != rec.get("sha256"):
            out.append(f"input {key} {rec.get('path')} has sha256 {got[:12]}..., the provenance "
                       f"{str(rec.get('sha256'))[:12]}...")
    if fp.provenance["inputs"].get("levels_day3") != {"path": LEVELS_DAY3,
                                                       "sha256": fp.provenance["day3"].get("levels_sha256")}:
        out.append("inputs.levels_day3 does not name Day 3's levels file and the sha256 day3 records")
    return out


def grammar_problems(pal: Palette) -> list[str]:
    """(g) the LoopScript regex is REGEX_CHARS long with REGEX_SHA256; (h) prompt v2 gives V2_REFERENCES references
    with strict=True."""
    from lofi_rl import prompt_v2
    from lofi_rl.grammar import loopscript_regex
    from lofi_rl.prompt import PromptError
    out = []
    rx = loopscript_regex(pal)
    if (len(rx), sha256_bytes(rx.encode("utf-8"))) != (REGEX_CHARS, REGEX_SHA256):
        out.append(f"the regex is {len(rx)} characters, sha256 {sha256_bytes(rx.encode('utf-8'))}; the frozen grammar "
                   f"is {REGEX_CHARS}, {REGEX_SHA256}")
    try:
        refs = prompt_v2.reference_texts(pal, strict=True)
        if len(refs) != V2_REFERENCES:
            out.append(f"prompt v2 gives {len(refs)} references, not {V2_REFERENCES}")
    except PromptError as err:
        out.append(f"prompt v2 refuses a reference: {err}")
    return out


def frozen_problems(fp: FrozenPalette, *, sealed: bool = False, gate_text: str | None = None) -> list[str]:
    """(i) `frozen` is null; with `sealed`, it is the date of Shresth's APPROVED line in the gate file."""
    from lofi_rl.freeze.gate import GateError, approval_date
    got = fp.palette.frozen
    if not sealed:
        return [] if got is None else [f"frozen is {got!r}: check a sealed file with --sealed"]
    try:
        date = approval_date(gate_text or "")
    except GateError as err:
        return [str(err)]
    if date is None:
        return ["no APPROVED line under the gate file's ## Approval: the file cannot be sealed"]
    return [] if got == date else [f"frozen is {got!r}, the APPROVED line's date is {date}"]


def offline_problems(fp: FrozenPalette, root: Path = ROOT) -> list[str]:
    """(b), (d)-(h): every check on committed files alone; (i) is `frozen_problems`."""
    out = inputs_problems(fp, root)
    try:
        dp = DP.load(root / fp.provenance["inputs"]["draft"]["path"])
        dev = yaml.safe_load((root / DEV.relative_to(ROOT)).read_text(encoding="utf-8"))
    except (DraftPaletteError, OSError, yaml.YAMLError) as err:
        return out + [str(err)]
    try:
        if rebuild_text(fp, root=root) != fp.text:
            out.append(f"{label(fp.path)} is not what build writes from its inputs (Day 3's levels from the "
                       "provenance)")
    except FreezeError as err:
        out.append(f"rebuild: {err}")
    out += picks_problems(fp)
    try:
        out += numbers_problems(fp, dp, dev, root)
        out += fold_problems(fp, dp)
    except (FreezeError, KeyError, TypeError, AttributeError) as err:
        out.append(f"{type(err).__name__}: {err}")
    out += grammar_problems(fp.palette)
    return out


def levels_problems(fp: FrozenPalette, day1: Path, day2: Path, day3: Path) -> list[str]:
    """With the three levels files: their sha256s are the ones recorded; every provenance number equals its value
    there (each instrument's t and F in both jobs, job 2's instrument, the K2 spread F, S*, the targets, and what Day 3
    measured); a fresh draft build is the draft (drafts.palette.build); a fresh build with this Day-3 file is the
    file."""
    out: list[str] = []
    prov = fp.provenance
    lv = prov["draft_levels"]
    docs = {}
    for name, path, want in (("day1", day1, lv["from"]["day1"]["sha256"]), ("day2", day2, lv["from"]["day2"]["sha256"]),
                             ("day3", day3, prov["day3"]["levels_sha256"])):
        doc, sha = _json(path, f"--levels-{name}")
        docs[name] = doc
        if sha != want:
            out.append(f"--levels-{name} {label(path)} has sha256 {sha[:12]}..., the provenance {want[:12]}...")
    if out:
        return out
    l1, l2, l3 = docs["day1"], docs["day2"], docs["day3"]
    for name, doc in docs.items():
        if doc.get("S_star") != lv["S_star"] or doc.get("targets") != lv["targets"]:
            out.append(f"{name}: S* {doc.get('S_star')} or targets {doc.get('targets')} are not the record's")
    if l3.get("moved") != prov["day3"]["moved"]:
        out.append(f"day3: moved {l3.get('moved')}, the provenance {prov['day3']['moved']}")
    recs = _records(lv)
    for where, option in band_options():
        rec = recs.get(where) or {}
        m = rec.get("measured") or {}
        for name, doc in (("day1", l1), ("day2", l2)):
            v = (doc.get("variants") or {}).get(option) or {}
            if {"t": v.get("t"), "F": v.get("F")} != m.get(name):
                out.append(f"{where}: the record's {name} {m.get(name)} is not {name}'s t {v.get('t')} F {v.get('F')}")
        v2 = (l2.get("variants") or {}).get(option) or {}
        what = m.get("as") or {}
        if not what or any(v2.get(k) != what[k] for k in what):
            out.append(f"{where}: the record's instrument {what} is not what job 2 measured")
        if where.startswith("keys."):
            cfg = ((l2.get("configs") or {}).get(f"k2_{option}") or {}).get("buses", {}).get("keys", {})
            if (cfg.get("loops") or {}).get("F") != rec.get("spread_F") or cfg.get("instrument") != option:
                out.append(f"{where}: spread_F {rec.get('spread_F')} is not job 2's K2 median "
                           f"{(cfg.get('loops') or {}).get('F')}")
        v3 = (l3.get("variants") or {}).get(option)
        got3 = (prov["day3"].get("measured") or {}).get(option)
        if (v3 is None) != (got3 is None) or (v3 is not None and {"t": v3.get("t"), "F": v3.get("F")} != got3):
            out.append(f"{where}: the provenance's Day-3 measure {got3} is not day3's "
                       f"{None if v3 is None else {'t': v3.get('t'), 'F': v3.get('F')}}")
    try:
        if DP.build(day1, day2) != DP.load(ROOT / prov["inputs"]["draft"]["path"]).text:
            out.append(f"{prov['inputs']['draft']['path']} is not what drafts.palette build writes from these files")
    except (DraftPaletteError, OSError, ValueError, KeyError) as err:
        out.append(f"draft rebuild: {err}")
    try:
        if rebuild_text(fp, levels_day3=day3) != fp.text:
            out.append(f"{label(fp.path)} is not what build writes with --levels-day3 {label(day3)}")
    except FreezeError as err:
        out.append(f"rebuild with --levels-day3: {err}")
    return out


def check(path: Path | str = FROZEN, *, gate: Path | str | None = None,
          levels: tuple[Path, Path, Path] | None = None, verify: bool = False, sealed: bool = False,
          root: Path = ROOT) -> tuple[FrozenPalette, list[str], list[str]]:
    """(the file, its problems, the checks run). FreezeError when the file does not load (a)."""
    from lofi_rl.freeze import gate as G
    fp = load(path)
    done = [f"(a) parses, provenance {PROVENANCE_SCHEMA}"]
    bad = offline_problems(fp, root)
    done += ["(b) rebuild byte-identical (Day 3 from the provenance)", "(d) picks", "(e) numbers traced",
             f"(f) folds {len(fp.doc['keys']) * len(fp.doc['kits'])} of {len(fp.doc['keys']) * len(fp.doc['kits'])}",
             f"(g) regex {REGEX_CHARS} chars sha256 {REGEX_SHA256}", f"(h) prompt v2 {V2_REFERENCES} of "
             f"{V2_REFERENCES} strict"]
    gate_text = None
    if gate is not None or sealed:
        gate_text = G.read(gate if gate is not None else GATE)
    bad += frozen_problems(fp, sealed=sealed, gate_text=gate_text)
    done.append(f"(i) frozen {fp.palette.frozen or 'null'}" + (" = the APPROVED line's date" if sealed else ""))
    if gate is not None:
        bad += G.picks_problems(gate_text) + G.constants_problems()
        done.append(f"(c) picks = {label(gate)} (Days 1-3, keep3)")
    if levels is not None:
        bad += levels_problems(fp, *levels)
        done.append("levels " + ", ".join(f"{label(p)}" for p in levels) + " equal; draft rebuild byte-identical; "
                    "rebuild with --levels-day3 byte-identical")
    if verify and not bad:
        try:
            load_palette(fp.path, verify=True, root=root)
        except PaletteError as err:
            bad.append(f"verify: {err}")
        done.append("sources, presets and coverage verified")
    return fp, bad, done


# --------------------------------------------------------------------------------------------- vocabulary
def vocabulary(fp: FrozenPalette) -> dict:
    """Everything a loop may write with the frozen palette (cf. drafts.palette.vocabulary): names and the shootout
    option each is, chords (and the chords whose every tone fits each mood's mode), voicings, lanes, edit targets,
    and the vamp ranks as seed data."""
    from lofi_rl.prompt import mood_chords, vocabulary as prompt_vocab
    v = prompt_vocab(fp.palette)
    lv = fp.provenance["draft_levels"]
    return {"keys": {k: lv["keys"][k]["option"] for k in fp.doc["keys"]},
            "kits": {k: {"kit": lv["kits"][k]["option"], "bass": lv["kits"][k]["bass"]["option"]}
                     for k in fp.doc["kits"]},
            "sax": {k: lv["sax"][k]["option"] for k in fp.doc["sax"]}, "numerals": list(v["numerals"]),
            "qualities": list(v["qualities"]), "voicings": list(v["voicings"]), "drum_lanes": list(LANES),
            "edit_targets": list(EDIT_TARGETS),
            "chords_in_mode": {m: list(c) for m, c in mood_chords(fp.palette).items()},
            "vamp_ranks": {m: [[{"vamp": n, "chords": c, "span": s} for n, c, s in tie] for tie in ranks]
                           for m, ranks in DP.VAMP_RANKS.items()}}


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.freeze", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write palettes/lofi_rain_v1.yaml from its inputs")
    c = sub.add_parser("check", help="validate the frozen palette")
    v = sub.add_parser("vocab", help="print what a loop may write with the palette")
    b.add_argument("--draft", type=Path, default=DRAFT)
    b.add_argument("--levels-day3", type=Path, required=True)
    b.add_argument("--plan", type=Path, default=PLAN)
    b.add_argument("--rain", type=Path, default=RAIN)
    b.add_argument("--frozen", default="null", help="null, or the date of Shresth's APPROVED line")
    b.add_argument("--out", type=Path, default=FROZEN)
    for sp in (c, v):
        sp.add_argument("--palette", type=Path, default=FROZEN)
    c.add_argument("--gate", type=Path, default=None, help="also check the picks against gates/GATE_palette.md")
    for day in ("day1", "day2", "day3"):
        c.add_argument(f"--levels-{day}", type=Path)
    c.add_argument("--verify", action="store_true", help="hash every source, read every preset and its coverage")
    c.add_argument("--sealed", action="store_true", help="frozen must be the APPROVED line's date (--gate or "
                                                         "gates/GATE_palette.md)")
    v.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    from lofi_rl.freeze.gate import GateError
    try:
        if a.cmd == "build":
            text = build(a.levels_day3, draft=a.draft, plan=a.plan, rain=a.rain,
                         frozen=None if a.frozen == "null" else a.frozen)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(text, encoding="utf-8")
            print(f"wrote {label(a.out)} (sha256 {sha256_bytes(text.encode('utf-8'))})")
            return 0
        if a.cmd == "vocab":
            voc = vocabulary(load(a.palette))
            if a.json:
                print(json.dumps(voc, indent=1))
            else:
                for k, val in voc.items():
                    print(f"{k}: {json.dumps(val)}")
            return 0
        lv = (a.levels_day1, a.levels_day2, a.levels_day3)
        if any(lv) and not all(lv):
            raise FreezeError("--levels-day1, --levels-day2 and --levels-day3 go together")
        fp, bad, done = check(a.palette, gate=a.gate, levels=lv if all(lv) else None, verify=a.verify,
                              sealed=a.sealed)
    except (FreezeError, GateError, DraftPaletteError, PaletteError, OSError, ValueError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    for x in bad:
        print(f"problem: {x}", file=sys.stderr)
    if bad:
        return 3
    print(f"ok: {label(fp.path)} ({fp.palette.id}, sha256 {fp.sha256}): " + "; ".join(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
