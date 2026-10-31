"""Level matching for the shootouts (docs/SHOOTOUTS.md §3 steps 1-6; §7.1 "Tool"; CD-25). No palette field yet:
the trims and faders it writes are carried by the palette generator as CD-51 says.

    python -m lofi_rl.shootout.levels --plan shootouts/plan.yaml --loops shootouts/loops \\
        --palettes <run>/summary/palettes/measure --stage day1 [--prior <levels.json>] \\
        --out <run>/summary/levels.json [--workers 8]

For every instrument variant i of the stage and every loop l of `--loops` (its briefs.yaml; the 9 reference loops):
1. stems: i's bus input with trim 0, from the measuring palettes (`lofi_rl.shootout.palettes --levels none`): the
   FluidSynth stem, or the drum sampler's cycle. One run renders one variant of each bus at once (8 runs on Day 1);
2. P(i, l), the integrated loudness of the bus input over the kept cycle, read 40 dB up so BS.1770's -70 LUFS gate
   does not bias a quiet input (input_lufs). t(i) = median_l P(g, l) - median_l P(i, l),
   g the level anchor of i's bus (plan.yaml `levels.anchors`); every instrument of `levels.zero_trim_source` (GU)
   gets t = 0;
3. L(i, l): the bus after its chain with the trim, before the fader;
4. S* = median_l L(sax target, l);
5. F(i) = S* + T(bus) - median_l L(i, l), T the base palette's mix.bus_target_lu. t and F are rounded to 0.01 dB, as
   mix.calibrated_gains rounds; `statistics.median` takes the mean of the middle two for an even count;
6. checks: (a) linear range: feeding the trimmed input 6.02 dB lower lowers L by 6.02 +- 0.1 dB (median over the
   loops), else the variant is flagged; (b) after the fader, the median bus loudness is S* + T within 0.05 LU, and
   each loop's is L(i, l) + F(i) within 0.1 LU (a pure gain moves a quiet bus by a few hundredths of an LU through
   BS.1770's -70 LUFS gate).
Stage level only: stems and bus chains, never the master chain, bed or loudness loop. The trims are applied exactly
as the palettes carry them: a SoundFont instrument's through the Gain trim slot heading its chain, a kit's on every
lane's dB, so L is the pre-fader loudness its clips' render manifests will report.

Drift keys (§3): the mirror stamp (<repo>/.sync-stamp), the base palette's sha256, each (run, loop)'s compiled digest,
and each variant's source and kit-map sha256. With --prior, equal keys reuse the prior values; any difference
re-measures every variant, and the prior keys, values and the moves are recorded (a move above 0.1 dB is listed).

Exit codes: 0; 2 for a bad argument; 3 when the inputs are refused (a missing variant, a failed gate, drift in a
measuring palette). Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
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
from lofi_rl.shootout.plan import PlanError, load_plan, read_briefs

SCHEMA = "lofi-rl/shootout-levels/1"
MEASURE_SCHEMA = "lofi-rl/shootout-measure/1"
SR = 44100
CYCLES = 2                                  # two cycles through the chains, keep the second (RENDER.md §2)
BUSES = ("sax", "keys", "bass", "drums")
TRIM_BUSES = ("sax", "keys", "bass")
STAGES = ("day1", "day2", "day3")
BUILT = ("day1",)
MOVE_DB = 0.1                               # a trim or fader that moves more is listed (§3, drift)
INPUT_LIFT_DB = 40.0                        # P is read this far up, then taken off: see input_lufs
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
               F: Mapping[str, float], target: Mapping[str, float], median_tol: float = 0.05,
               loop_tol: float = 0.1) -> dict[str, dict]:
    """§3 step 6b: after the fader, the median bus loudness is S* + T within `median_tol` LU, and every loop's is its
    own L + F within `loop_tol` LU."""
    out = {}
    for v, row in post.items():
        median = _median(row, v)
        worst = max(abs(row[k] - (L[v][k] + F[v])) for k in row)
        out[v] = {"lufs": dict(row), "median_lufs": median, "median_off_lu": median - target[v],
                  "worst_loop_lu": worst, "ok": abs(median - target[v]) <= median_tol and worst <= loop_tol}
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
    name: str
    text: str
    brief: Brief


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
            for case, stems in zip(cases, ex.map(one, cases)):
                self._stems[(run.id, case.name)] = stems

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
               prior: Path | str | None = None, workers: int = 8, bench: Bench | None = None,
               stamp: str | None = None, cache: Path | str | None = None, root: Path = ROOT) -> dict:
    """§3 steps 1-6 for the stage's variants; returns the levels document (see the module docstring)."""
    if stage not in BUILT:
        raise LevelsError(f"stage {stage}: only {', '.join(BUILT)} is built (job 1)")
    plan = load_plan(plan_path, root=root)
    measure_dir = Path(measure_dir)
    index_bytes = (measure_dir / "index.json").read_bytes()
    index = json.loads(index_bytes)
    if index.get("schema") != MEASURE_SCHEMA or index.get("stage") != stage:
        raise LevelsError(f"{measure_dir}/index.json: expected {MEASURE_SCHEMA} for {stage}")
    lv = plan.levels
    variants = index["variants"]
    day = STAGES.index(stage) + 1
    wanted = [o for st in plan.stages_of(day) for o in st.options if o not in index.get("without", [])]
    for o in wanted:
        if o not in variants:
            raise LevelsError(f"{measure_dir}/index.json does not measure {o}, a day-{day} option")
    for vid, v in variants.items():
        if v["flags"].get("voicing") != "rootless" or v["flags"].get("sax_warm") or not v["flags"].get("sax_expr"):
            raise LevelsError(f"{vid}: Day-2 variant flags {v['flags']} are built for job 2")
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
    runs = [replace(r, compiled=MappingProxyType({c.name: compiled_case(r.palette, c) for c in cases})) for r in runs]
    stamp = read_stamp(root) if stamp is None else stamp
    drift = {"stamp": stamp, "base_sha256": index["base_sha256"],
             "digests": {r.id: {c.name: r.compiled[c.name].digest() for c in cases} for r in runs},
             "sources": {vid: v["source_sha256"] for vid, v in sorted(variants.items())},
             "kit_maps": {vid: v["kit_map_sha256"] for vid, v in sorted(variants.items()) if v["kit_map_sha256"]}}
    targets = {b: float(runs[0].palette.mix.bus_target_lu[b]) for b in BUSES}
    doc = {"schema": SCHEMA, "stage": stage, "plan": str(plan_path), "plan_sha256": plan.sha256,
           "base": index["base"], "base_sha256": index["base_sha256"],
           "measure": {"dir": str(measure_dir), "index_sha256": hashlib.sha256(index_bytes).hexdigest()},
           "loops": {c.name: {"brief": {"mood": c.brief.mood, "energy": c.brief.energy, "bpm": c.brief.bpm,
                                        "tonic": c.brief.tonic},
                              "n": runs[0].compiled[c.name].n,
                              "text_sha256": hashlib.sha256(c.text.encode("utf-8")).hexdigest()} for c in cases},
           "without": list(index.get("without", [])), "anchors": dict(lv.anchors), "sax_target": lv.sax_target,
           "zero_trim_source": lv.zero_trim_source, "targets": targets,
           "tolerances": {"linear_drop_db": lv.linear_drop_db, "linear_tol_db": lv.linear_tol_db,
                          "median_tol_lu": lv.median_tol_lu, "loop_tol_lu": lv.loop_tol_lu},
           "input_lift_db": INPUT_LIFT_DB,
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
        if not changed:
            doc.update(variants=prior_doc["variants"], S_star=prior_doc["S_star"], flags=prior_doc["flags"],
                       reused=True)
            return doc

    bench = bench if bench is not None else AudioBench(workers=workers, cache=cache, root=root)
    bus_of = {vid: v["bus"] for vid, v in variants.items()}
    zero = {vid for vid, v in variants.items() if v["source"] == lv.zero_trim_source}
    P, L, L_low, post, t_all, F_all = {}, {}, {}, {}, {}, {}
    s_star = None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for run in runs:
            new = [(b, run.instruments[b]) for b in run.measure]
            if not new:
                continue
            bench.prepare(run, cases, [b for b, _ in new])
            for b, vid in new:
                P[vid] = {c.name: bench.input_lufs(run, c, b) for c in cases}
            t = trims({k: P[k] for k in P if k in {vid for _, vid in new} | set(lv.anchors.values())},
                      bus_of, lv.anchors, zero)
            jobs = [(b, vid, c, trim) for b, vid in new for c in cases
                    for trim in (t[vid], t[vid] - lv.linear_drop_db)]
            levels = list(ex.map(lambda j: bench.chain_lufs(run, j[2], j[0], j[3], keep=j[3] == t[j[1]]), jobs))
            for (b, vid, c, trim), level in zip(jobs, levels):
                (L if trim == t[vid] else L_low).setdefault(vid, {})[c.name] = level
            if s_star is None:
                s_star = sax_target(L, lv.sax_target)
            F = faders({vid: L[vid] for _, vid in new}, bus_of, s_star, targets)
            for b, vid in new:
                t_all[vid], F_all[vid] = t[vid], F[vid]
                post[vid] = {c.name: bench.post_lufs(run, c, b, F[vid]) for c in cases}
            bench.release(run)
    checks = post_check(post, L, F_all, {v: s_star + targets[bus_of[v]] for v in post}, lv.median_tol_lu,
                        lv.loop_tol_lu)
    out = {}
    for vid, v in sorted(variants.items()):
        lin = linear_check(L[vid], L_low[vid], lv.linear_drop_db, lv.linear_tol_db)
        rec = {**v, "P": P[vid], "L": L[vid], "L_low": L_low[vid], "t": t_all[vid], "F": F_all[vid],
               "S_star": s_star, "T": targets[v["bus"]],
               "checks": {"linear": lin, "post_fader": checks[vid]},
               "drift": {"stamp": stamp, "base_sha256": index["base_sha256"], "digests": drift["digests"][v["run"]]}}
        if prior_doc is not None and vid in prior_doc.get("variants", {}):
            old = prior_doc["variants"][vid]
            rec["prior"] = {"t": old["t"], "F": old["F"]}
            rec["moved_db"] = {"t": round(rec["t"] - old["t"], 2), "F": round(rec["F"] - old["F"], 2)}
            if max(abs(x) for x in rec["moved_db"].values()) > MOVE_DB:
                doc["moved"].append(vid)
        out[vid] = rec
    doc.update(variants=out, S_star=s_star,
               flags=sorted(vid for vid, r in out.items()
                            if not (r["checks"]["linear"]["ok"] and r["checks"]["post_fader"]["ok"])))
    return doc


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None, *, bench: Bench | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.shootout.levels", description=__doc__.split("\n\n")[0])
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--loops", required=True, type=Path, help="a folder of loops and their briefs.yaml")
    p.add_argument("--palettes", required=True, type=Path, help="the measuring palettes: <run>/summary/palettes/measure")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--prior", type=Path, default=None, help="an earlier levels.json: re-measure only on drift")
    p.add_argument("--out", required=True, type=Path, help="<run>/summary/levels.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", type=Path, default=None, help="one-shot cache (default $LOFI_CACHE or ~/.cache/lofi-rl)")
    a = p.parse_args(argv)
    if a.stage not in BUILT:
        p.error(f"--stage {a.stage}: only {', '.join(BUILT)} is built so far (job 1)")
    for name, path in (("--plan", a.plan), ("--loops", a.loops / "briefs.yaml"),
                       ("--palettes", a.palettes / "index.json"), ("--prior", a.prior)):
        if path is not None and not path.is_file():
            p.error(f"{name}: {path} is not a file")
    try:
        doc = run_levels(a.plan, a.loops, a.palettes, a.stage, prior=a.prior, workers=a.workers, bench=bench,
                         cache=a.cache)
    except (LevelsError, PlanError) as err:
        print(f"levels: {err}", file=sys.stderr)
        return 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"S* = {doc['S_star']:.3f} LUFS ({doc['sax_target']}); targets " +
          ", ".join(f"{b} {t:+.1f}" for b, t in doc["targets"].items()) +
          (" (reused: no drift)" if doc["reused"] else ""))
    for vid, r in doc["variants"].items():
        lin, post = r["checks"]["linear"], r["checks"]["post_fader"]
        print(f"{vid:22s} {r['bus']:5s} t {r['t']:+7.2f}  F {r['F']:+7.2f}  P~ {statistics.median(r['P'].values()):7.2f}"
              f"  L~ {statistics.median(r['L'].values()):7.2f}  drop {lin['median_drop_db']:5.2f}"
              f" {'ok' if lin['ok'] else 'FLAG'}  post {post['median_off_lu']:+.3f}/{post['worst_loop_lu']:.3f}"
              f" {'ok' if post['ok'] else 'FLAG'}")
    for vid in doc["moved"]:
        print(f"moved > {MOVE_DB} dB: {vid} {doc['variants'][vid]['moved_db']}")
    print(f"wrote {a.out}" + (f"; flagged: {', '.join(doc['flags'])}" if doc["flags"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
