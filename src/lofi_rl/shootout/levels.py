"""Level matching for the shootouts (docs/SHOOTOUTS.md §3 steps 1-6; §7.1 "Tool"; CD-25, CD-52). No palette field
yet: the trims and faders it writes are carried by the palette generator as CD-51 says.

    python -m lofi_rl.shootout.levels --plan shootouts/plan.yaml --loops shootouts/loops \\
        --palettes <run>/summary/palettes/measure --stage day1|day2|day3 [--draw <run>/summary/draw.json] \\
        [--prior <levels.json>] --out <run>/summary/levels.json [--workers 8]

For every instrument variant i of the stage and every loop l of `--loops` (its briefs.yaml; the 9 reference loops):
1. stems: i's bus input with trim 0, from the measuring palettes (`lofi_rl.shootout.palettes --levels none`): the
   FluidSynth stem, or the drum sampler's cycle. One run renders one variant of each bus at once (8 runs on Day 1);
2. P(i, l), the integrated loudness of the bus input over the kept cycle, read 40 dB up so BS.1770's -70 LUFS gate
   does not bias a quiet input (input_lufs). t(i) = median_l P(g, l) - median_l P(i, l),
   g the level anchor of i's bus (plan.yaml `levels.anchors`); every instrument of `levels.zero_trim_source` (GU)
   gets t = 0;
3. L(i, l): the bus after its chain with the trim, before the fader;
4. S* = median_l L(sax target, l);
5. F(i) = S* + T(bus) - median_l L(i, l), T the base palette's mix.bus_target_lu: the median fader, which the frozen
   palette takes. Each shootout clip takes its own fader instead (CD-52): F(i, l) = S* + T(bus) - L(i, l), then
   corrected by the loudness the bus measures after that fader until it is within CLIP_FIX_LU of S* + T, at most
   CLIP_STEPS times (`clip_fader`; a pure gain moves a quiet bus through BS.1770's -70 LUFS gate). Trims and faders
   are rounded to 0.01 dB, as mix.calibrated_gains rounds; `statistics.median` takes the mean of the middle two for
   an even count;
6. checks: (a) linear range: feeding the trimmed input 6.02 dB lower lowers L by 6.02 +- 0.1 dB (median over the
   loops), else the variant is flagged; (b) at the median fader, the median bus loudness is S* + T within 0.05 LU
   (how far each loop moved from its own L + F through the gate is kept as `worst_loop_lu`); (c) at its own fader,
   every clip's bus loudness is S* + T within clip_tol_lu (0.1 LU).
Stage level only: stems and bus chains, never the master chain, bed or loudness loop. The trims are applied exactly
as the palettes carry them: a SoundFont instrument's through the Gain trim slot heading its chain, a kit's on every
lane's dB, so L is the pre-fader loudness its clips' render manifests will report, and the loudness after a fader is
that of the very signal a render sums into its master (mix.mix_buses scales the bus in float64, as here).

Cases: the clip faders cover every (variant, loop) measured above. With `--draw` (the job's draw.json), every clip
the draw makes whose palette names a `level` (lofi_rl.shootout.draw, CD-52) must be one of those, or its case joins
the measure plan: a clip whose text is not its loop's own (an anchor's sour copy, case `<loop>.sour`) is measured on
that text, once, in its candidate's run, with the candidate's trim. Extra cases never enter P, t, the median F or
check (a).

Drift keys (§3): the mirror stamp (<repo>/.sync-stamp), the base palette's sha256, each (run, case)'s compiled digest,
and each variant's source and kit-map sha256. With --prior, equal keys reuse the prior values; any difference
re-measures every variant, and the prior keys, values and the moves are recorded (a move above 0.1 dB is listed).

Day 2 on (CD-53; `--draw` required): the variants are every instrument the draw plays, measured as above (the stamp
changes at every sync and the runs differ from job 1's, so with job 1's levels.json as --prior they are always
measured again, each with its move from job 1 recorded; a prior of another day is never reused). Then every stage
palette a matched clip plays (the measuring index's `configs`) is measured on each case its clips play, plus the 9
reference loops for a `loops` stage (K2 in spread voicing, S2a), on each bus its stage matches, with that bus's
instrument's trim: L, the clip's own fader (`clip_fader`), the loudness after it, and the loudness at the band's
median fader. `configs` holds them; `rounds` holds every drawn round's spread on each matched bus at the median faders
(`before_lu`) and at the clips' own (`after_lu`).

Day 3 (section 4.5) runs as Day 2: every instrument its draw plays (the Day-2 picks in every rain palette, and the
S2 re-run's sax when the job has one) measured again against job 1's levels.json, so the rain palettes carry each
one's t and median F, the values the frozen palette takes. The rain design rounds match nothing (section 3 "Not
matched": the bed is set by loudness under each clip's own music, RENDER.md section 3.4), and the repeats play their
first showing's renders and are not measured. The anchor's sour copy is matched to its round-mates (2026-11-01
review: at the median F its sax sat up to 1.65 LU from theirs): its rain palette is a config (`target: round`)
measured on the sax at the band sax's trim on the loop's own text, case `<loop>`, the round-mates' sax L(loop) at
the median F, and on the sour text, case `<loop>.sour`, whose own fader starts at F + L(loop) - L(loop.sour) and is
corrected (`clip_fader`) until its sax measures within CLIP_FIX_LU of L(loop) + F. The palette, not the instrument
run, since the band's sax variant and timing play there. An S2 re-run adds its configs as Day 2's.

Exit codes: 0; 2 for a bad argument; 3 when the inputs are refused (a missing variant, a failed gate, drift in a
measuring palette, a draw of another plan or with a clip on a loop `--loops` does not hold). Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import yaml

from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import CompileError, compile_loop
from lofi_rl.compile.events import CompiledLoop
from lofi_rl.dsl import parse
from lofi_rl.palette import ROOT, Palette, PaletteError, Plugin, parse_palette, sf2_presets
from lofi_rl.render.loudness import lufs
from lofi_rl.render.mix import bus_chain, process
from lofi_rl.render.sampler import KitShots, kit_shots, mix_hits
from lofi_rl.shootout.plan import SLOT_BUS, PlanError, load_plan, read_briefs, substitute

SCHEMA = "lofi-rl/shootout-levels/1"
MEASURE_SCHEMA = "lofi-rl/shootout-measure/1"
DRAW_SCHEMA = "lofi-rl/shootout-draw/1"
SR = 44100
CYCLES = 2                                  # two cycles through the chains, keep the second (RENDER.md §2)
BUSES = ("sax", "keys", "bass", "drums")
TRIM_BUSES = ("sax", "keys", "bass")
STAGES = ("day1", "day2", "day3")
BUILT = ("day1", "day2", "day3")
MOVE_DB = 0.1                               # a trim or fader that moves more is listed (§3, drift)
INPUT_LIFT_DB = 40.0                        # P is read this far up, then taken off: see input_lufs
CLIP_FIX_LU = 0.005                         # a clip's fader is corrected until it measures this close (half 0.01 dB)
CLIP_STEPS = 3                              # at most this many corrections of a clip's fader
_EPS = 1e-9                                 # float noise in a measured miss of exactly half a rounding step
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class LevelsError(ValueError):
    pass


# ------------------------------------------------------------------------------------- §3 steps 2-6
def _median(row: Mapping[str, float], what: str) -> float:
    vals = list(row.values())
    if not vals or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        raise LevelsError(f"{what}: needs finite loudness values on every loop, got {dict(row)}")
    return statistics.median(vals)


def trims(P: Mapping[str, Mapping[str, float]], bus_of: Mapping[str, str], anchors: Mapping[str, str],
          zero: Iterable[str] = ()) -> dict[str, float]:
    """§3 step 2: t(i) = median P(g) - median P(i), g the anchor of i's bus, rounded to 0.01 dB; `zero` gets 0."""
    zero = set(zero)
    med = {v: _median(row, v) for v, row in P.items()}
    for v in P:
        g = anchors.get(bus_of[v])
        if g not in med:
            raise LevelsError(f"{v}: the {bus_of[v]} anchor {g} was not measured, so its trim has no reference")
    return {v: 0.0 if v in zero else round(med[anchors[bus_of[v]]] - med[v], 2) for v in P}


def sax_target(L: Mapping[str, Mapping[str, float]], anchor: str) -> float:
    """§3 step 4: S* = the median over the loops of the named anchor's L."""
    if anchor not in L:
        raise LevelsError(f"the sax target {anchor} was not measured")
    return _median(L[anchor], anchor)


def faders(L: Mapping[str, Mapping[str, float]], bus_of: Mapping[str, str], s_star: float,
           targets: Mapping[str, float]) -> dict[str, float]:
    """§3 step 5: F(i) = S* + T(bus) - median L(i), rounded to 0.01 dB."""
    return {v: round(s_star + targets[bus_of[v]] - _median(row, v), 2) for v, row in L.items()}


def linear_check(L: Mapping[str, float], L_low: Mapping[str, float], drop: float = 6.02,
                 tol: float = 0.1) -> dict:
    """§3 step 6a: feeding the trimmed input `drop` dB lower must lower L by `drop` +- `tol` (median over loops)."""
    drops = {k: L[k] - L_low[k] for k in L}
    median = _median(drops, "linear-range drop")
    return {"drop_db": drops, "median_drop_db": median, "ok": abs(median - drop) <= tol}


def post_check(post: Mapping[str, Mapping[str, float]], L: Mapping[str, Mapping[str, float]],
               F: Mapping[str, float], target: Mapping[str, float], median_tol: float = 0.05) -> dict[str, dict]:
    """§3 step 6b: at the median fader F (the frozen palette's), the median bus loudness is S* + T within
    `median_tol` LU. `worst_loop_lu`, how far a loop's loudness moved from its own L + F through BS.1770's -70 LUFS
    gate, is information: a shootout clip plays its own fader, measured after it (step 6c, CD-52)."""
    out = {}
    for v, row in post.items():
        median = _median(row, v)
        worst = max(abs(row[k] - (L[v][k] + F[v])) for k in row)
        out[v] = {"lufs": dict(row), "median_lufs": median, "median_off_lu": median - target[v],
                  "worst_loop_lu": worst, "ok": abs(median - target[v]) <= median_tol}
    return out


def clip_fader(L: float, target: float, post_at: Callable[[float], float], *, fix: float = CLIP_FIX_LU,
               steps: int = CLIP_STEPS) -> dict:
    """§3 step 5 for one clip (CD-52): F = target - L rounded to 0.01 dB, the loudness `post_at(F)` measured after it,
    and F corrected by the miss until it is within `fix` LU, at most `steps` times. A pure gain moves a quiet bus
    through BS.1770's -70 LUFS gate (the Day-1 dry run's kit_virtuosity drum bus on L1, -55.0 LUFS before its fader,
    measured 0.101 LU under L + F), so L + F is not what the bus measures. Of the faders tried, the one measured
    closest to the target is kept (the lower one on a tie)."""
    if not (math.isfinite(L) and math.isfinite(target)):
        raise LevelsError(f"a bus at {L} LUFS after its chain is silent: no fader can bring it to {target:.2f}")
    F = round(target - L, 2)
    tried: dict[float, float] = {}
    for _ in range(1 + steps):
        tried[F] = float(post_at(F))
        miss = target - tried[F]
        if abs(miss) <= fix + _EPS:
            break
        nxt = round(F + miss, 2)
        if nxt in tried:
            break
        F = nxt
    best = min(tried, key=lambda f: (abs(tried[f] - target), f))
    return {"L": L, "F": best, "lufs": tried[best], "target_lufs": target, "off_lu": tried[best] - target,
            "tried": len(tried)}


def round_fader(L: Mapping[str, float], case: "Case", F: float, post_at: Callable[[float], float]) -> dict:
    """Day 3's rain anchor (2026-11-01 review): a case's fader against its round-mates rather than S* + T. The
    round-mates play the loop's own text at the band's median fader F, so the loop's case keeps F and its target is
    its own L + F; a sour copy (`<loop>.sour`) starts at F + L(loop) - L(loop.sour) and is corrected by `clip_fader`
    until its bus measures within CLIP_FIX_LU of L(loop) + F, the round-mates' level. `L` holds the bus's L per case
    name; the record adds `target: round` and the `reference` case."""
    ref = case.loop if case.variant != "base" else case.name
    target = L[ref] + F
    if case.name == ref:
        got = float(post_at(F))
        rec = {"L": L[ref], "F": F, "lufs": got, "target_lufs": target, "off_lu": got - target, "tried": 1}
    else:
        rec = clip_fader(L[case.name], target, post_at)
    return {**rec, "target": "round", "reference": ref}


def clip_check(clips: Mapping[str, Mapping[str, Mapping]], tol: float = 0.1) -> dict[str, dict]:
    """§3 step 6c: every clip, at its own fader, sits at S* + T within `tol` LU. `off` lists the cases that do not."""
    out = {}
    for v, row in clips.items():
        worst = max(abs(c["off_lu"]) for c in row.values())
        out[v] = {"worst_off_lu": worst, "cases": len(row),
                  "off": sorted(k for k, c in row.items() if abs(c["off_lu"]) > tol), "ok": worst <= tol}
    return out


# ------------------------------------------------------------------------------------- chains and trims
def input_lufs(x: np.ndarray, lift_db: float = INPUT_LIFT_DB) -> float:
    """P, the integrated loudness of a bus input (§3 step 2), read `lift_db` up and taken back down.

    BS.1770's -70 LUFS absolute gate is its only part that depends on the level, and inputs sit at very different
    levels: the VSCO 2 pizz bass stem peaks near -65 dBFS on L4 (the GU bass, -27). Read plain (2026-10-31, the 9
    reference loops), its loudness is -inf on L4 and L6, and up to 7.37 LU too high on the others, where the gate keeps
    only its loudest blocks; the sparse GU drums read 0.38 LU high on L1. 40 dB up, the gate sits at -110 LUFS of the
    input and drops only near-silence, so two inputs compare alike however quiet either is. On the other GU anchors'
    inputs the two readings differ by 0.02 LU at most. L and the post-fader levels stay plain BS.1770, as the render
    manifest measures them."""
    level = float(lufs(np.ascontiguousarray(x, dtype=np.float64) * 10.0 ** (lift_db / 20.0), SR)) - lift_db
    if not math.isfinite(level):
        raise LevelsError(f"the bus input is silent, or under {-70 - lift_db:.0f} LUFS: no trim can match it")
    return level


def with_trim(chain: Sequence[Plugin], trim: float) -> tuple[Plugin, ...]:
    """A pitched bus's chain with its trim slot (the leading [Gain, {gain_db}], CD-51) set to `trim` dB."""
    if not chain or chain[0].name != "Gain" or set(chain[0].params) != {"gain_db"}:
        raise LevelsError("the chain has no trim slot: a shootout palette's sax, keys and bass chains start with "
                          "[Gain, {gain_db: trim}] (shootouts/palettes/base.yaml)")
    return (Plugin("Gain", MappingProxyType({"gain_db": float(trim)})),) + tuple(chain[1:])


def chain_lufs(chain: Sequence[Plugin], stem: np.ndarray, trim: float, n: int) -> tuple[float, np.ndarray]:
    """L of a pitched bus: its stem (all cycles) through the chain with the trim; the loudness of the last cycle."""
    out = process(with_trim(chain, trim), stem, SR)
    last = np.ascontiguousarray(out[:, -n:])
    return float(lufs(last, SR)), last


def drum_stem(compiled: CompiledLoop, kit: KitShots, trim: float, cycles: int = CYCLES) -> np.ndarray:
    """The drum stem with the kit's trim on every lane's dB, exactly as a palette carries it (CD-51)."""
    lanes = {lane: (pan, db + trim) for lane, (pan, db) in kit.lanes.items()}
    return np.tile(mix_hits(compiled.drums, kit.shots, lanes, compiled.n), (1, cycles))


# ------------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Case:
    name: str                               # a reference loop's name, or `<loop>.<variant>` for a drawn text (CD-52)
    text: str
    brief: Brief
    loop: str | None = None                 # an extra case: the reference loop whose text it changes
    variant: str = "base"                   # an extra case: the draw's render variant (`sour`)


@dataclass(frozen=True, eq=False)
class Run:
    id: str
    file: str
    sha256: str
    palette: Palette
    instruments: Mapping[str, str] = field(hash=False)
    measure: tuple[str, ...] = ()
    compiled: Mapping[str, CompiledLoop] = field(default_factory=dict, hash=False)


_VERIFIED: dict[tuple, str | None] = {}          # one hash per source file and process (§7.7)
_COMPILED: dict[tuple, CompiledLoop] = {}        # (palette sha256, loop text, brief) -> its CompiledLoop


def _verify(pal: Palette, root: Path) -> None:
    problems = []
    for sid, src in sorted(pal.sources.items()):
        path = pal.source_path(sid, root)
        try:
            st = path.stat()
        except OSError:
            problems.append(f"source {sid}: {path} is missing")
            continue
        key = (str(path), st.st_size, st.st_mtime_ns, st.st_ino, src.sha256)
        if key not in _VERIFIED:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            _VERIFIED[key] = None if h.hexdigest() == src.sha256 else f"source {sid}: {path} does not match its sha256"
        if _VERIFIED[key]:
            problems.append(_VERIFIED[key])
    if not problems:
        found: dict[str, frozenset] = {}
        for where, src, bank, prog in pal.presets():
            if src not in found:
                found[src] = sf2_presets(pal.source_path(src, root))
            if (bank, prog) not in found[src]:
                problems.append(f"{where}: bank {bank} prog {prog} is not a preset of source {src}")
    if problems:
        raise LevelsError(f"palette {pal.id}: " + "; ".join(problems))


def load_run_palette(path: Path, sha256: str, root: Path) -> Palette:
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise LevelsError(f"{path} changed after the generator wrote it (sha256 {got[:12]}..., index says "
                          f"{sha256[:12]}...)")
    try:
        pal = parse_palette(yaml.load(data.decode("utf-8"), Loader=_LOADER), str(path), got)
    except (PaletteError, yaml.YAMLError) as err:
        raise LevelsError(f"{path}: {err}") from err
    _verify(pal, root)
    return pal


def compiled_case(pal: Palette, case: Case) -> CompiledLoop:
    key = (pal.sha256, case.text, case.brief)
    if key not in _COMPILED:
        res = parse(case.text, vocab=pal.vocab(), mode=case.brief.mode)
        if not res.gate.ok:
            raise LevelsError(f"loop {case.name} fails gate {res.gate.code} under {pal.id}: {res.gate.detail}")
        try:
            _COMPILED[key] = compile_loop(res.ast, case.brief, pal)
        except CompileError as err:
            raise LevelsError(f"loop {case.name} does not compile under {pal.id}: {err}") from err
    return _COMPILED[key]


def read_cases(loops_dir: Path) -> list[Case]:
    try:
        briefs = read_briefs(loops_dir)
    except (OSError, PlanError) as err:
        raise LevelsError(f"--loops {loops_dir}: {err}") from err
    return [Case(name, (Path(loops_dir) / f"{name}.ls").read_text(encoding="utf-8"), b) for name, b in briefs.items()]


def read_stamp(root: Path) -> str | None:
    try:
        return (Path(root) / ".sync-stamp").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


@dataclass(frozen=True)
class DrawCases:
    """What the job's draw adds to the measure plan (CD-52): the extra cases each variant plays, and the record."""

    by_variant: Mapping[str, tuple[Case, ...]]
    cases: Mapping[str, Case]               # case name -> the extra case
    variants: Mapping[str, tuple[str, ...]]  # case name -> the variants measured on it
    file: dict


def _brief_doc(b: Brief) -> dict:
    return {"mood": b.mood, "energy": b.energy, "bpm": b.bpm, "tonic": b.tonic}


def draw_cases(path: Path | str, plan_sha256: str, stage: str, cases: Sequence[Case],
               variants: Mapping[str, Mapping]) -> DrawCases:
    """Every clip of the job's draw.json whose palette names a `level` (lofi_rl.shootout.draw: the bus, the variant
    and the case its fader is measured on) must play a loop of `cases`. A clip of its loop's own text is measured
    with that loop already; any other text (an anchor's sour copy, case `<loop>.sour`) becomes an extra case of its
    variant. Raises LevelsError for a draw of another plan or stage, a clip on a loop `cases` does not hold, a clip
    of a variant the measuring palettes do not measure on its bus, or one case drawn with two texts."""
    data = Path(path).read_bytes()
    try:
        doc = json.loads(data)
    except ValueError as err:
        raise LevelsError(f"--draw {path}: {err}") from err
    if not isinstance(doc, dict) or doc.get("schema") != DRAW_SCHEMA or doc.get("stage") != stage:
        raise LevelsError(f"--draw {path}: expected a {DRAW_SCHEMA} draw of {stage}")
    made = str((doc.get("plan") or {}).get("sha256"))
    if made != plan_sha256:
        raise LevelsError(f"--draw {path} was made from another plan (sha256 {made[:12]}...), not "
                          f"{plan_sha256[:12]}...")
    by_name = {c.name: c for c in cases}
    extra: dict[str, Case] = {}
    on: dict[str, list[str]] = {}
    palettes = doc.get("palettes") or {}
    for rid, entry in sorted((doc.get("renders") or {}).items()):
        want = (palettes.get(entry.get("palette")) or {}).get("level")
        if not want or "buses" in want:             # a Day-2 clip is measured on its own palette (config_cases)
            continue
        vid, case, bus, loop = want["variant"], want["case"], want["bus"], entry["loop"]
        if (variants.get(vid) or {}).get("bus") != bus:
            raise LevelsError(f"--draw: clip {rid} takes the {bus} fader of {vid}, which the measuring palettes do "
                              f"not measure on that bus")
        if loop not in by_name:
            raise LevelsError(f"--draw: clip {rid} plays {loop}, which --loops does not hold, so its fader cannot "
                              f"be measured")
        base = by_name[loop]
        if entry.get("brief") != _brief_doc(base.brief):
            raise LevelsError(f"--draw: clip {rid} has brief {entry.get('brief')}, not {loop}'s")
        if case == loop:
            if entry["text"] != base.text:
                raise LevelsError(f"--draw: clip {rid} names case {loop} but plays another text than --loops holds")
            continue
        if case != f"{loop}.{entry['variant']}":
            raise LevelsError(f"--draw: clip {rid}'s case {case!r} is neither {loop} nor {loop}.{entry['variant']}")
        x = Case(case, entry["text"], base.brief, loop, entry["variant"])
        if extra.setdefault(case, x) != x:
            raise LevelsError(f"--draw: case {case} is drawn with two different texts")
        if vid not in on.setdefault(case, []):
            on[case].append(vid)
    by_variant: dict[str, list[Case]] = {}
    for case, vids in on.items():
        for vid in vids:
            by_variant.setdefault(vid, []).append(extra[case])
    return DrawCases(MappingProxyType({v: tuple(xs) for v, xs in by_variant.items()}), MappingProxyType(extra),
                     MappingProxyType({c: tuple(v) for c, v in on.items()}),
                     {"file": str(path), "sha256": hashlib.sha256(data).hexdigest()})


def config_cases(draw_doc: Mapping, configs: Mapping[str, Mapping], cases: Sequence[Case]) -> dict[str, list[Case]]:
    """Day 2 on (CD-53): the cases each stage palette (`configs`, from the measuring index) is measured on. Every
    clip of the draw whose palette names a Day-2 `level` (its buses and its case) adds its own text on its base;
    a `loops` config adds the 9 reference loops with its stage's literal substitution (K2's `keys: ep spread`).
    Case names are those of the draw: the loop, or `<loop>.<variant>` for a changed text (`spread`, `sour`, a vamp
    id). LevelsError for a clip on a palette the index does not measure, on a loop `cases` does not hold, with
    another brief, or for one case name with two texts."""
    by_name = {c.name: c for c in cases}
    out: dict[str, dict[str, Case]] = {k: {} for k in configs}
    palettes = draw_doc.get("palettes") or {}
    for rid, entry in sorted((draw_doc.get("renders") or {}).items()):
        spec = palettes.get(entry.get("palette")) or {}
        level = spec.get("level") or {}
        if "buses" not in level:
            continue
        base, loop = spec.get("base"), entry["loop"]
        if base not in configs:
            raise LevelsError(f"--draw: clip {rid} plays stage palette {base}, which the measuring index does not "
                              f"measure")
        if list(level["buses"]) != list(configs[base]["buses"]):
            raise LevelsError(f"--draw: clip {rid} matches {level['buses']}, its palette {base} "
                              f"{configs[base]['buses']}")
        if loop not in by_name:
            raise LevelsError(f"--draw: clip {rid} plays {loop}, which --loops does not hold, so its faders cannot "
                              f"be measured")
        if entry.get("brief") != _brief_doc(by_name[loop].brief):
            raise LevelsError(f"--draw: clip {rid} has brief {entry.get('brief')}, not {loop}'s")
        want = loop if entry["variant"] == "base" else f"{loop}.{entry['variant']}"
        if level["case"] != want:
            raise LevelsError(f"--draw: clip {rid}'s case {level['case']!r} is not {want!r}")
        x = Case(want, entry["text"], by_name[loop].brief, loop, entry["variant"])
        if out[base].setdefault(want, x) != x:
            raise LevelsError(f"--draw: case {want} of {base} is drawn with two different texts")
        if level.get("target") == "round":      # Day 3's rain anchor: its round-mates' text on the same palette
            mates = [e for e in (draw_doc.get("renders") or {}).values()
                     if e.get("palette") == base and e["loop"] == loop]
            if configs[base].get("target") != "round" or not mates or any(
                    e["text"] != by_name[loop].text for e in mates):
                raise LevelsError(f"--draw: clip {rid} is matched to its round on {base}, but no clip of its round "
                                  f"plays {base} with {loop}'s own text, or the index does not match it so")
            ref = Case(loop, by_name[loop].text, by_name[loop].brief, loop, "base")
            if out[base].setdefault(loop, ref) != ref:
                raise LevelsError(f"--draw: case {loop} of {base} is drawn with two different texts")
    for key, cfg in configs.items():
        if not cfg.get("loops"):
            continue
        sub = dict(cfg.get("sub") or {})
        variant = sub["keys"].split()[-1] if "keys" in sub else "base"
        for c in cases:
            name = c.name if variant == "base" else f"{c.name}.{variant}"
            x = Case(name, substitute(c.text, **sub) if sub else c.text, c.brief, c.name, variant)
            if out[key].setdefault(name, x) != x:
                raise LevelsError(f"{key}: case {name} is drawn with another text than its loop's substitution")
    return {k: [v[n] for n in sorted(v)] for k, v in out.items()}


def loop_case_names(cfg: Mapping, cases: Sequence[Case]) -> list[str]:
    """A `loops` config's 9 reference-loop case names: the loop's name, or `<loop>.spread` under K2's substitution."""
    sub = dict(cfg.get("sub") or {})
    variant = sub["keys"].split()[-1] if "keys" in sub else "base"
    return [c.name if variant == "base" else f"{c.name}.{variant}" for c in cases]


def measure_configs(configs: Mapping[str, Mapping], measure_dir: Path, cases_of: Mapping[str, Sequence[Case]],
                    cases: Sequence[Case], P: Mapping[str, Mapping[str, float]], t_all: Mapping[str, float],
                    F_all: Mapping[str, float], s_star: float, targets: Mapping[str, float], lv, bench: "Bench",
                    ex: ThreadPoolExecutor, root: Path) -> dict:
    """Day 2 on (CD-53): every stage palette a matched clip plays, measured on each of its cases on each bus it
    matches, with that bus's instrument's trim (measured above: an option never gets a trim of its own, since the
    frozen palette holds one trim per instrument, CD-28): L, the clip's own fader (`clip_fader`), the loudness after
    it, and the loudness at the band's median fader (`median_lufs`, what the clip would play without its own). A
    `loops` config also gets §3 steps 3-6 on the 9 reference loops: its median fader F (the frozen palette's spread
    and sax-variant values), the linear-range and median checks, and its input's median shift from its
    instrument's (`input_shift_db`, information)."""
    out = {}
    for key, cfg in configs.items():
        pal = load_run_palette(Path(measure_dir) / cfg["file"], cfg["sha256"], root)
        cs = list(cases_of[key])
        buses = list(cfg["buses"])
        inst = {b: cfg["instruments"][b] for b in buses}
        missing = sorted(b for b, i in inst.items() if i not in t_all)
        if missing:
            raise LevelsError(f"{key}: its {missing} instruments were not measured, so they have no trim")
        run = Run(f"config:{key}", cfg["file"], cfg["sha256"], pal, MappingProxyType(dict(cfg["instruments"])),
                  tuple(buses))
        run = replace(run, compiled=MappingProxyType({c.name: compiled_case(pal, c) for c in cs}))
        nine = set(loop_case_names(cfg, cases)) if cfg.get("loops") else set()
        bench.prepare(run, cs, buses)
        jobs = [(b, c, t_all[inst[b]]) for b in buses for c in cs]
        jobs += [(b, c, t_all[inst[b]] - lv.linear_drop_db) for b in buses for c in cs if c.name in nine]
        levels = list(ex.map(lambda j: bench.chain_lufs(run, j[1], j[0], j[2], keep=j[2] == t_all[inst[j[0]]]),
                             jobs))
        L: dict[str, dict[str, float]] = {b: {} for b in buses}
        L_low: dict[str, dict[str, float]] = {b: {} for b in buses}
        for (b, c, trim), level in zip(jobs, levels):
            (L if trim == t_all[inst[b]] else L_low)[b][c.name] = level
        pairs = [(b, c) for b in buses for c in cs]
        med = list(ex.map(lambda j: bench.post_lufs(run, j[1], j[0], F_all[inst[j[0]]]), pairs))
        if cfg.get("target") == "round":        # Day 3's rain anchor: the sour copy at its round-mates' sax
            done = list(ex.map(lambda j: round_fader(L[j[0]], j[1], F_all[inst[j[0]]],
                                                     lambda f, j=j: bench.post_lufs(run, j[1], j[0], f)), pairs))
        else:
            done = list(ex.map(lambda j: clip_fader(L[j[0]][j[1].name], s_star + targets[j[0]],
                                                    lambda f, j=j: bench.post_lufs(run, j[1], j[0], f)), pairs))
        P_in = {b: {c.name: bench.input_lufs(run, c, b) for c in cs if c.name in nine} for b in buses} if nine else {}
        rec = {}
        for b in buses:
            target = s_star + targets[b]
            cases_rec = {}
            for (bb, c), m, clip in zip(pairs, med, done):
                if bb == b:             # off at the median fader: from S* + T, or from the round-mates' level
                    cases_rec[c.name] = {**clip, "lu": clip["lufs"] - s_star, "instrument": inst[b],
                                         "t": t_all[inst[b]], "median_F": F_all[inst[b]], "median_lufs": m,
                                         "median_off_lu": m - (clip["target_lufs"] if "reference" in clip else target)}
            entry = {"instrument": inst[b], "t": t_all[inst[b]], "median_F": F_all[inst[b]], "target_lufs": target,
                     "cases": cases_rec, "checks": {"clips": clip_check({b: cases_rec}, lv.clip_tol_lu)[b]}}
            if cfg.get("target") == "round":    # each case's own target: its round-mates' L + the median F
                entry.update(target_lufs=None, target="round")
            if nine:
                L9 = {n: L[b][n] for n in sorted(nine)}
                F9 = round(target - _median(L9, f"{key}:{b}"), 2)
                by = {c.name: c for c in cs}
                post9 = dict(zip(sorted(nine), ex.map(lambda n: bench.post_lufs(run, by[n], b, F9), sorted(nine))))
                entry["loops"] = {
                    "P": P_in[b], "L": L9, "L_low": {n: L_low[b][n] for n in sorted(nine)}, "F": F9,
                    "input_shift_db": _median(P_in[b], key) - _median(P[inst[b]], inst[b]),
                    "checks": {"linear": linear_check(L9, {n: L_low[b][n] for n in L9}, lv.linear_drop_db,
                                                      lv.linear_tol_db),
                               "post_fader": post_check({b: post9}, {b: L9}, {b: F9}, {b: target},
                                                        lv.median_tol_lu)[b]}}
            rec[b] = entry
        out[key] = {"stage": cfg["stage"], "option": cfg["option"], "file": cfg["file"], "sha256": cfg["sha256"],
                    "instruments": dict(cfg["instruments"]), "settings": list(cfg["settings"]),
                    "sax_muted": cfg["sax_muted"], "loops": bool(cfg.get("loops")), "buses": rec,
                    **({"target": "round"} if cfg.get("target") == "round" else {}),
                    "cases": {c.name: {"loop": c.loop, "variant": c.variant, "n": run.compiled[c.name].n,
                                       "text_sha256": hashlib.sha256(c.text.encode("utf-8")).hexdigest()} for c in cs},
                    "digests": {c.name: run.compiled[c.name].digest() for c in cs}}
        bench.release(run)
    return out


def config_flags(configs: Mapping[str, Mapping]) -> list[str]:
    """`<config>:<bus>` for every matched bus whose clips or 9-loop checks fail."""
    out = []
    for key, cfg in configs.items():
        for b, entry in cfg["buses"].items():
            checks = [entry["checks"]["clips"]] + list((entry.get("loops") or {}).get("checks", {}).values())
            if not all(c["ok"] for c in checks):
                out.append(f"{key}:{b}")
    return sorted(out)


def round_spreads(draw_doc: Mapping, configs: Mapping[str, Mapping]) -> list[dict]:
    """Each drawn round's spread on every bus its clips match (§3 step 8): the largest minus the smallest loudness
    after the fader, at the band's median faders (`before_lu`, what the clips would play without their own) and at
    each clip's own (`after_lu`), with the clips' cases and options. A Day-3 repeat shows its first showing's renders,
    which this draw does not make, so it has no spread here (its hidden manifest carries the first showing's). A Day-3
    rain anchor round (`round_spread`) gives its sour copy's sax against its round-mates'."""
    palettes, out = draw_doc.get("palettes") or {}, []
    for b in draw_doc.get("batches") or []:
        for r in b["rounds"]:
            rows = []
            for c in r["clips"]:
                entry = draw_doc["renders"].get(c["render"])
                if r["kind"] == "repeat" or entry is None or entry.get("palette_path"):
                    break
                spec = palettes.get(entry["palette"]) or {}
                level = spec.get("level") or {}
                if "buses" not in level:
                    break
                rows.append((configs[spec["base"]], level))
            if len(rows) != len(r["clips"]) or not rows:
                matched = round_spread(draw_doc, configs, b["name"], r)
                if matched:
                    out.append(matched)
                continue
            per_bus = {}
            for bus in rows[0][1]["buses"]:
                recs = [cfg["buses"][bus]["cases"][lvl["case"]] for cfg, lvl in rows]
                before = [x["median_lufs"] for x in recs]
                after = [x["lufs"] for x in recs]
                per_bus[bus] = {"before_lu": max(before) - min(before), "after_lu": max(after) - min(after)}
            out.append({"batch": b["name"], "stage": r["stage"], "slot": r["slot"], "kind": r["kind"],
                        "loop": r["loop"], "options": [c["option"] for c in r["clips"]], "buses": per_bus,
                        "before_lu": max(v["before_lu"] for v in per_bus.values()),
                        "after_lu": max(v["after_lu"] for v in per_bus.values())})
    return out


def round_spread(draw_doc: Mapping, configs: Mapping[str, Mapping], batch: str, r: Mapping) -> dict | None:
    """Day 3's rain anchor round: the sax of its sour copy (the clip whose level matches its round) against its
    round-mates', which play the loop's own text at the band's median fader on palettes that differ from the sour
    copy's base only in their bed: at the median faders (`before_lu`) and at the sour copy's own (`after_lu`). None
    for any other round."""
    palettes = draw_doc.get("palettes") or {}
    for c in r["clips"] if r["kind"] == "anchor" else ():
        spec = palettes.get((draw_doc["renders"].get(c["render"]) or {}).get("palette")) or {}
        level = spec.get("level") or {}
        if level.get("target") != "round":
            continue
        per_bus = {}
        for bus in level["buses"]:
            cases = configs[spec["base"]]["buses"][bus]["cases"]
            sour = cases[level["case"]]
            ref = cases[sour["reference"]]
            per_bus[bus] = {"before_lu": abs(sour["median_lufs"] - ref["median_lufs"]),
                            "after_lu": abs(sour["lufs"] - ref["lufs"])}
        return {"batch": batch, "stage": r["stage"], "slot": r["slot"], "kind": r["kind"], "loop": r["loop"],
                "options": [x["option"] for x in r["clips"]], "buses": per_bus, "target": "round",
                "before_lu": max(v["before_lu"] for v in per_bus.values()),
                "after_lu": max(v["after_lu"] for v in per_bus.values())}
    return None


# ------------------------------------------------------------------------------------------ benches
class Bench(Protocol):
    """What the tool measures through. AudioBench renders; tests pass a table of made-up values."""

    def prepare(self, run: Run, cases: Sequence[Case], buses: Sequence[str]) -> None: ...
    def input_lufs(self, run: Run, case: Case, bus: str) -> float: ...
    def chain_lufs(self, run: Run, case: Case, bus: str, trim: float, keep: bool = False) -> float: ...
    def post_lufs(self, run: Run, case: Case, bus: str, fader: float) -> float: ...
    def release(self, run: Run) -> None: ...


class AudioBench:
    """Stems from the renderer's own stages: pipeline.fluid_stems for keys, bass and sax (only the buses a run
    measures), the drum sampler for the kit; bus chains from render/mix.py, BS.1770 from render/loudness.py."""

    def __init__(self, workers: int = 4, cache: Path | str | None = None, root: Path = ROOT):
        self.workers, self.cache, self.root = max(1, workers), cache, root
        self._stems: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._kits: dict[str, KitShots] = {}
        self._kept: dict[tuple[str, str, str], np.ndarray] = {}

    def prepare(self, run, cases, buses):
        from lofi_rl.render.pipeline import fluid_stems      # imports FluidSynth's wrapper only when rendering
        pitched = [b for b in buses if b in TRIM_BUSES]
        if "drums" in buses:
            self._kits[run.id] = kit_shots(run.palette, next(iter(run.palette.kits)), cache=self.cache,
                                           root=self.root, workers=self.workers)

        def one(case: Case) -> dict[str, np.ndarray]:
            c = run.compiled[case.name]
            if not pitched:
                return {}
            c = replace(c, **{s: replace(getattr(c, s), notes=()) for s in TRIM_BUSES if s not in pitched})
            stems = fluid_stems(c, run.palette, CYCLES, root=self.root, workers=1)
            return {b: stems[b] for b in pitched}
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for case, stems in zip(cases, ex.map(one, cases)):           # a case staged again adds its buses
                self._stems.setdefault((run.id, case.name), {}).update(stems)

    def _stem(self, run: Run, case: Case, bus: str, trim: float) -> np.ndarray:
        if bus == "drums":
            return drum_stem(run.compiled[case.name], self._kits[run.id], trim)
        return self._stems[(run.id, case.name)][bus]

    def input_lufs(self, run, case, bus):
        n = run.compiled[case.name].n
        try:
            return input_lufs(self._stem(run, case, bus, 0.0)[:, -n:])
        except LevelsError as err:
            raise LevelsError(f"{run.instruments[bus]} on {case.name} ({run.id}): {err}") from err

    def chain_lufs(self, run, case, bus, trim, keep=False):
        n = run.compiled[case.name].n
        chain = bus_chain(run.palette.mix, bus)
        if bus == "drums":
            out = np.ascontiguousarray(process(chain, self._stem(run, case, bus, trim), SR)[:, -n:])
            level = float(lufs(out, SR))
        else:
            level, out = chain_lufs(chain, self._stem(run, case, bus, 0.0), trim, n)
        if keep:
            self._kept[(run.id, case.name, bus)] = out
        return level

    def post_lufs(self, run, case, bus, fader):
        out = self._kept[(run.id, case.name, bus)]
        return float(lufs(out.astype(np.float64) * 10.0 ** (fader / 20.0), SR))

    def release(self, run):
        for d in (self._stems, self._kept):
            for k in [k for k in d if k[0] == run.id]:
                del d[k]
        self._kits.pop(run.id, None)


# ------------------------------------------------------------------------------------------- the tool
def _flat(d: Mapping, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def changed_keys(old: Mapping, new: Mapping) -> list[str]:
    a, b = _flat(old), _flat(new)
    return sorted(k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>"))


def run_levels(plan_path: Path | str, loops_dir: Path | str, measure_dir: Path | str, stage: str = "day1", *,
               draw: Path | str | None = None, prior: Path | str | None = None, workers: int = 8,
               bench: Bench | None = None, stamp: str | None = None, cache: Path | str | None = None,
               root: Path = ROOT) -> dict:
    """§3 steps 1-6 for the stage's variants, and each clip's own fader (CD-52) on every (variant, case) measured:
    the reference loops, and with `draw` every other text a drawn clip plays. Returns the levels document (see the
    module docstring)."""
    if stage not in BUILT:
        raise LevelsError(f"stage {stage}: only {', '.join(BUILT)} is built")
    plan = load_plan(plan_path, root=root)
    measure_dir = Path(measure_dir)
    index_bytes = (measure_dir / "index.json").read_bytes()
    index = json.loads(index_bytes)
    if index.get("schema") != MEASURE_SCHEMA or index.get("stage") != stage:
        raise LevelsError(f"{measure_dir}/index.json: expected {MEASURE_SCHEMA} for {stage}")
    lv = plan.levels
    variants = index["variants"]
    day = STAGES.index(stage) + 1
    draw_doc = None
    if day == 1:
        wanted = [o for st in plan.stages_of(day) for o in st.options if o not in index.get("without", [])]
        what = f"a day-{day} option"
    else:                           # Day 2 on (CD-53): every instrument the draw plays, and each palette it matches
        if draw is None:
            raise LevelsError(f"stage {stage}: the level tool needs the job's draw (--draw): its clips and band")
        draw_bytes = Path(draw).read_bytes()
        if (index.get("draw") or {}).get("sha256") != hashlib.sha256(draw_bytes).hexdigest():
            raise LevelsError(f"{measure_dir}/index.json was made from another draw than {draw}")
        draw_doc = json.loads(draw_bytes)
        wanted = sorted(set(lv.anchors.values()) | {spec["band"][s] for spec in draw_doc["palettes"].values()
                                                     for s in SLOT_BUS if spec["band"].get(s)})
        what = f"an instrument the day-{day} draw plays"
    for o in wanted:
        if o not in variants:
            raise LevelsError(f"{measure_dir}/index.json does not measure {o}, {what}")
    for vid, v in variants.items():
        if v["flags"].get("voicing") != "rootless" or v["flags"].get("sax_warm") or not v["flags"].get("sax_expr"):
            raise LevelsError(f"{vid}: flags {v['flags']} are not an instrument's own; a Day-2 option is measured on "
                              f"the stage palette its clips play (CD-53)")
    runs = []
    for r in index["runs"]:
        runs.append(Run(r["run"], r["file"], r["sha256"], load_run_palette(measure_dir / r["file"], r["sha256"], root),
                        MappingProxyType(dict(r["instruments"])), tuple(r["measure"])))
    first = {vid: v["run"] for vid, v in variants.items()}
    for vid, v in variants.items():
        r = next((x for x in runs if x.id == v["run"]), None)
        if r is None or r.instruments[v["bus"]] != vid or v["bus"] not in r.measure:
            raise LevelsError(f"{vid}: run {v['run']} does not measure it on bus {v['bus']}")
    measured = [(r.instruments[b], r.id) for r in runs for b in r.measure]
    if sorted(measured) != sorted(first.items()):
        raise LevelsError(f"{measure_dir}/index.json: every variant must be measured in exactly one run")
    anchor_run = runs[0].id
    for bus, g in lv.anchors.items():
        if first.get(g) != anchor_run:
            raise LevelsError(f"the {bus} anchor {g} must be measured in the first run, {anchor_run}")
    if first.get(lv.sax_target) != anchor_run:
        raise LevelsError(f"the sax target {lv.sax_target} must be measured in the first run, {anchor_run}")

    cases = read_cases(Path(loops_dir))
    drawn = (draw_cases(draw, plan.sha256, stage, cases, variants) if draw is not None
             else DrawCases(MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), {}))

    def extras(r: Run) -> list[Case]:              # the drawn texts of the variants a run measures, once each
        return list({x.name: x for b in r.measure for x in drawn.by_variant.get(r.instruments[b], ())}.values())

    runs = [replace(r, compiled=MappingProxyType({c.name: compiled_case(r.palette, c) for c in [*cases, *extras(r)]}))
            for r in runs]
    stamp = read_stamp(root) if stamp is None else stamp
    drift = {"stamp": stamp, "base_sha256": index["base_sha256"],
             "digests": {r.id: {name: c.digest() for name, c in r.compiled.items()} for r in runs},
             "sources": {vid: v["source_sha256"] for vid, v in sorted(variants.items())},
             "kit_maps": {vid: v["kit_map_sha256"] for vid, v in sorted(variants.items()) if v["kit_map_sha256"]}}
    targets = {b: float(runs[0].palette.mix.bus_target_lu[b]) for b in BUSES}
    n_of = {name: c.n for r in runs for name, c in r.compiled.items()}
    doc = {"schema": SCHEMA, "stage": stage, "plan": str(plan_path), "plan_sha256": plan.sha256,
           "base": index["base"], "base_sha256": index["base_sha256"],
           "measure": {"dir": str(measure_dir), "index_sha256": hashlib.sha256(index_bytes).hexdigest()},
           "loops": {c.name: {"brief": _brief_doc(c.brief), "n": runs[0].compiled[c.name].n,
                              "text_sha256": hashlib.sha256(c.text.encode("utf-8")).hexdigest()} for c in cases},
           "draw": drawn.file or None,
           "extra_cases": {name: {"loop": x.loop, "variant": x.variant, "variants": list(drawn.variants[name]),
                                  "text_sha256": hashlib.sha256(x.text.encode("utf-8")).hexdigest(),
                                  "brief": _brief_doc(x.brief), "n": n_of[name]}
                           for name, x in sorted(drawn.cases.items())},
           "without": list(index.get("without", [])), "anchors": dict(lv.anchors), "sax_target": lv.sax_target,
           "zero_trim_source": lv.zero_trim_source, "targets": targets,
           "tolerances": {"linear_drop_db": lv.linear_drop_db, "linear_tol_db": lv.linear_tol_db,
                          "median_tol_lu": lv.median_tol_lu, "clip_tol_lu": lv.clip_tol_lu},
           "clip_fix_lu": CLIP_FIX_LU, "clip_steps": CLIP_STEPS, "input_lift_db": INPUT_LIFT_DB,
           "runs": [{"run": r.id, "file": r.file, "palette_sha256": r.sha256, "instruments": dict(r.instruments),
                     "measure": list(r.measure)} for r in runs],
           "drift": drift, "prior": None, "reused": False, "moved": []}

    prior_doc = None
    if prior is not None:
        prior_bytes = Path(prior).read_bytes()
        prior_doc = json.loads(prior_bytes)
        changed = changed_keys(prior_doc.get("drift", {}), drift)
        doc["prior"] = {"file": str(prior), "sha256": hashlib.sha256(prior_bytes).hexdigest(),
                        "drift": prior_doc.get("drift"), "changed": changed}
        if not changed and prior_doc.get("stage") == stage and draw_doc is None:   # Day 2 always measures its clips
            doc.update(variants=prior_doc["variants"], S_star=prior_doc["S_star"], flags=prior_doc["flags"],
                       reused=True)
            return doc

    bench = bench if bench is not None else AudioBench(workers=workers, cache=cache, root=root)
    bus_of = {vid: v["bus"] for vid, v in variants.items()}
    zero = {vid for vid, v in variants.items() if v["source"] == lv.zero_trim_source}
    P, L, L_low, L_x, post, t_all, F_all, clips = {}, {}, {}, {}, {}, {}, {}, {}
    s_star = None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for run in runs:
            new = [(b, run.instruments[b]) for b in run.measure]
            if not new:
                continue
            bench.prepare(run, cases, [b for b, _ in new])
            xs = [(b, vid, x) for b, vid in new for x in drawn.by_variant.get(vid, ())]
            for bus in dict.fromkeys(b for b, _, _ in xs):          # each drawn text once, on its variants' bus
                bench.prepare(run, list({x.name: x for b, _, x in xs if b == bus}.values()), [bus])
            for b, vid in new:
                P[vid] = {c.name: bench.input_lufs(run, c, b) for c in cases}
            t = trims({k: P[k] for k in P if k in {vid for _, vid in new} | set(lv.anchors.values())},
                      bus_of, lv.anchors, zero)
            jobs = [(b, vid, c, trim, L if trim == t[vid] else L_low) for b, vid in new for c in cases
                    for trim in (t[vid], t[vid] - lv.linear_drop_db)]
            jobs += [(b, vid, x, t[vid], L_x) for b, vid, x in xs]
            levels = list(ex.map(lambda j: bench.chain_lufs(run, j[2], j[0], j[3], keep=j[3] == t[j[1]]), jobs))
            for (b, vid, c, trim, table), level in zip(jobs, levels):
                table.setdefault(vid, {})[c.name] = level
            if s_star is None:
                s_star = sax_target(L, lv.sax_target)
            F = faders({vid: L[vid] for _, vid in new}, bus_of, s_star, targets)
            meds = [(b, vid, c) for b, vid in new for c in cases]
            for (b, vid, c), level in zip(meds, ex.map(lambda j: bench.post_lufs(run, j[2], j[0], F[j[1]]), meds)):
                post.setdefault(vid, {})[c.name] = level
            own = [(b, vid, c, L[vid][c.name]) for b, vid in new for c in cases]
            own += [(b, vid, x, L_x[vid][x.name]) for b, vid, x in xs]
            done = ex.map(lambda j: clip_fader(j[3], s_star + targets[j[0]],
                                               lambda f, j=j: bench.post_lufs(run, j[2], j[0], f)), own)
            for (b, vid, c, _), rec in zip(own, done):
                clips.setdefault(vid, {})[c.name] = {**rec, "lu": rec["lufs"] - s_star}
            for b, vid in new:
                t_all[vid], F_all[vid] = t[vid], F[vid]
            bench.release(run)
        if draw_doc is not None:        # Day 2 on: each matched clip on its own palette and text (CD-53)
            configs = index.get("configs") or {}
            cases_of = config_cases(draw_doc, configs, cases)
            doc["configs"] = measure_configs(configs, measure_dir, cases_of, cases, P, t_all, F_all, s_star, targets,
                                             lv, bench, ex, root)
            doc["rounds"] = round_spreads(draw_doc, doc["configs"])
    checks = post_check(post, L, F_all, {v: s_star + targets[bus_of[v]] for v in post}, lv.median_tol_lu)
    clip_checks = clip_check(clips, lv.clip_tol_lu)
    out = {}
    for vid, v in sorted(variants.items()):
        lin = linear_check(L[vid], L_low[vid], lv.linear_drop_db, lv.linear_tol_db)
        rec = {**v, "P": P[vid], "L": L[vid], "L_low": L_low[vid], "t": t_all[vid], "F": F_all[vid],
               "S_star": s_star, "T": targets[v["bus"]], "clips": clips[vid],
               "checks": {"linear": lin, "post_fader": checks[vid], "clips": clip_checks[vid]},
               "drift": {"stamp": stamp, "base_sha256": index["base_sha256"], "digests": drift["digests"][v["run"]]}}
        if prior_doc is not None and vid in prior_doc.get("variants", {}):
            old = prior_doc["variants"][vid]
            rec["prior"] = {"t": old["t"], "F": old["F"]}
            rec["moved_db"] = {"t": round(rec["t"] - old["t"], 2), "F": round(rec["F"] - old["F"], 2)}
            if max(abs(x) for x in rec["moved_db"].values()) > MOVE_DB:
                doc["moved"].append(vid)
        out[vid] = rec
    doc.update(variants=out, S_star=s_star,
               flags=sorted(vid for vid, r in out.items() if not all(c["ok"] for c in r["checks"].values()))
               + config_flags(doc.get("configs") or {}))
    return doc


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None, *, bench: Bench | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.shootout.levels", description=__doc__.split("\n\n")[0])
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--loops", required=True, type=Path, help="a folder of loops and their briefs.yaml")
    p.add_argument("--palettes", required=True, type=Path, help="the measuring palettes: <run>/summary/palettes/measure")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--draw", type=Path, default=None,
                   help="the job's draw.json: measure every text its level-matched clips play (CD-52)")
    p.add_argument("--prior", type=Path, default=None, help="an earlier levels.json: re-measure only on drift")
    p.add_argument("--out", required=True, type=Path, help="<run>/summary/levels.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", type=Path, default=None, help="one-shot cache (default $LOFI_CACHE or ~/.cache/lofi-rl)")
    a = p.parse_args(argv)
    if a.stage not in BUILT:
        p.error(f"--stage {a.stage}: only {', '.join(BUILT)} is built")
    for name, path in (("--plan", a.plan), ("--loops", a.loops / "briefs.yaml"),
                       ("--palettes", a.palettes / "index.json"), ("--draw", a.draw), ("--prior", a.prior)):
        if path is not None and not path.is_file():
            p.error(f"{name}: {path} is not a file")
    try:
        doc = run_levels(a.plan, a.loops, a.palettes, a.stage, draw=a.draw, prior=a.prior, workers=a.workers,
                         bench=bench, cache=a.cache)
    except (LevelsError, PlanError) as err:
        print(f"levels: {err}", file=sys.stderr)
        return 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"S* = {doc['S_star']:.3f} LUFS ({doc['sax_target']}); targets " +
          ", ".join(f"{b} {t:+.1f}" for b, t in doc["targets"].items()) +
          (" (reused: no drift)" if doc["reused"] else ""))
    for vid, r in doc["variants"].items():
        lin, post, clips = r["checks"]["linear"], r["checks"]["post_fader"], r["checks"]["clips"]
        own = [c["F"] for c in r["clips"].values()]
        print(f"{vid:22s} {r['bus']:5s} t {r['t']:+7.2f}  F {r['F']:+7.2f}  P~ {statistics.median(r['P'].values()):7.2f}"
              f"  L~ {statistics.median(r['L'].values()):7.2f}  drop {lin['median_drop_db']:5.2f}"
              f" {'ok' if lin['ok'] else 'FLAG'}  post {post['median_off_lu']:+.3f}/{post['worst_loop_lu']:.3f}"
              f" {'ok' if post['ok'] else 'FLAG'}  clips {len(own)} F {min(own):+.2f}..{max(own):+.2f}"
              f" off {clips['worst_off_lu']:.4f} {'ok' if clips['ok'] else 'FLAG'}")
    for name, x in doc["extra_cases"].items():
        print(f"drawn case {name}: measured for {', '.join(x['variants'])}")
    for key, cfg in (doc.get("configs") or {}).items():
        for b, e in cfg["buses"].items():
            own = [c["F"] for c in e["cases"].values()]
            nine = e.get("loops")
            print(f"config {key:28s} {b:5s} {e['instrument']:20s} t {e['t']:+7.2f}  median F {e['median_F']:+7.2f}"
                  f"  clips {len(own)} F {min(own):+.2f}..{max(own):+.2f} off {e['checks']['clips']['worst_off_lu']:.4f}"
                  + (f"  9 loops F {nine['F']:+.2f} drop {nine['checks']['linear']['median_drop_db']:.2f}"
                     f" input {nine['input_shift_db']:+.2f} dB" if nine else ""))
    for r in doc.get("rounds") or []:
        print(f"round {r['stage']:4s} {r['slot']:7s} {r['kind']:6s} {r['loop']:20s} spread before {r['before_lu']:.4f}"
              f" after {r['after_lu']:.4f} LU (" + ", ".join(f"{b} {v['before_lu']:.3f}/{v['after_lu']:.4f}"
                                                         for b, v in r["buses"].items()) + ")")
    for vid in doc["moved"]:
        print(f"moved > {MOVE_DB} dB: {vid} {doc['variants'][vid]['moved_db']}")
    print(f"wrote {a.out}" + (f"; flagged: {', '.join(doc['flags'])}" if doc["flags"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
