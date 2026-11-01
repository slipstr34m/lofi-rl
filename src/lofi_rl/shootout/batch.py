"""The shootout batch builder (docs/SHOOTOUTS.md section 7.6; docs/RATING.md Part 2).

Two steps, because the palettes and the renders depend on the draw (R2 plays each bass on the kit its round
draws, S1 leaves out the sources its rounds draw, an anchor needs its sour copy rendered):

    python -m lofi_rl.shootout.batch draw --plan shootouts/plan.yaml --stage day1|day2|day3 --summary <run>/summary
        [--leaders <slot>=<id>[+<id>...]/...] [--prior <run>/summary/batches.json ...]
        [--repeats <stage>+...] [--rerun-s2 <sax id>] [--cut <name>+...] [--seed <16 hex>]
    python -m lofi_rl.shootout.batch build --plan shootouts/plan.yaml --stage day1|day2|day3 --renders <run>/renders
        --out <inbox root> [--draw <run>/summary/draw.json] [--summary <run>/summary]

`--leaders` takes the job's forms, which differ by day (docs/SHOOTOUTS.md section 7.7); the draw refuses a slot, id or
count its day cannot play, as `draw.day_band` does.

`draw` draws from `secrets.randbits(64)` unless `--seed` is given (tests only) and writes `summary/draw.json`:
the maps, the batches' rounds in order with their hidden controls, and the renders and palettes the job must make.
The anchors' sour copies are written there and gate-checked (G1-G6 in the loop's mood, P_chrom >= 0.2).

`build` reads the draw and the job's renders and writes every batch of the stage as `<out>/<date>/<batch>/`:
one FLAC per clip appearance under an opaque name (`k` + 10 hex), `manifest.json` (public) and
`manifest.hidden.json` (what each clip really is: for a clip of an instrument stage, its own fader `fader_db` and
its bus's loudness as the level tool measured it after that fader, `bus_lufs`, and re S*, `bus_lu`; CD-52; from Day
2 on the stage's bus there, the vamps' keys, and every other bus the clip matches as `<bus>_fader_db`, `<bus>_lufs`
and `<bus>_lu`; CD-53), then
runs `python -m lofi_rl.ui.manifest check` on each folder and records the id map in `summary/batches.json`. Nothing
is written unless everything is in order: a missing render, a failed gate, a clip that is not exactly one pass of
its loop, a FLAC that changed since it was rendered, a repeat whose render is not the first showing's, a clip whose
draw palette names a `level` but whose record carries none for that case or failed it, an opaque id or note that
would leak a hidden value, or a batch folder that already exists, stops the build before the first file is
written. Batches are staged in `<out>/.staging-*` and moved into place only after every folder passes the check.

Opaque ids are drawn per batch from the draw's seed (so one seed gives byte-identical batches), and an id that
would contain any word or real-id stretch the leak check looks for (docs/RATING.md: "Leak check") is drawn again.

Sound claims: none.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets as _secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from lofi_rl.compile.timeline import loop_samples
from lofi_rl.ui.manifest import (BATCH_RE, HIDDEN_NAME, LEAK_WINDOW, PUBLIC_NAME, SCHEMA_VERSION, _words,
                                 audio_info, sha256_file)

from .draw import BARS, DrawError, Rng, draw, load_prior, parse_leaders, write_draw
from .plan import MIX_ORDER, PLAN, Plan, PlanError, load_plan

BATCHES_SCHEMA = "lofi-rl/shootout-batches/1"
SR = 44100
DURATION_TOL = 0.02             # RATING.md: one pass is bars x 240 / bpm seconds within 2 %
SRC = Path(__file__).resolve().parents[2]


class BuildError(ValueError):
    """The build stopped before writing anything. `errors` lists every problem found."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__(f"{len(self.errors)} problem(s), nothing written:\n  - " + "\n  - ".join(self.errors))


# ------------------------------------------------------------------------------------------------ leaks
def _secret_words(rounds: Mapping[str, Mapping]) -> set[str]:
    """What an opaque id must not contain, exactly as `ui.manifest.leak_errors` reads the hidden manifest: every
    word of every hidden clip value and brief id, and every 10-hex stretch of every real clip id."""
    out: set[str] = set()
    for hr in rounds.values():
        out |= _words(hr.get("brief_id") or "")
        for info in hr["clips"].values():
            real = str(info.get("clip", ""))[1:]
            out |= {real[k:k + LEAK_WINDOW] for k in range(len(real) - LEAK_WINDOW + 1)}
            for key, val in info.items():
                if key != "clip":
                    out |= _words(val)
    return out


def hidden_names(hidden: Mapping) -> set[str]:
    """The option, loop, palette, kit and render names a hidden manifest holds."""
    names: set[str] = set()
    for hr in hidden["rounds"].values():
        names.add(hr.get("brief_id") or "")
        for info in hr["clips"].values():
            names |= {str(info.get(k) or "") for k in ("option", "loop", "render", "kit")}
            names.add(str(info.get("palette") or "").split("@")[0])
    return names - {""}


def public_leaks(public: Mapping, hidden: Mapping, extra: Iterable[str] = ()) -> list[str]:
    """Hidden values that show in a public manifest (section 7.6 test 7). The names are the batch's own option,
    loop, palette, kit and render names (`hidden_names`) plus `extra` (the candidate lists its stages draw from):
    no word of one may appear in the public text (the batch name, scene, note, mode, round and clip ids and file
    names) unless it is a category word (ui/manifest.py's exemption: b1103-3-sax and its note may say "sax"), and
    no 10-hex stretch of a real clip id or a render's pcm_sha256 may appear in an id or a file name."""
    names = hidden_names(hidden) | set(extra)
    cats = _words(hidden.get("category") or "")
    for hr in hidden["rounds"].values():
        cats |= _words(hr.get("category") or "")
    ids = [x for r in public["rounds"] for x in [r["id"], *(v for c in r["clips"] for v in (c["id"], c["file"]))]]
    texts = [str(public.get(k) or "") for k in ("batch", "scene", "note", "mode")] + ids
    words = set().union(*(_words(t) for t in texts))
    out = []
    for n in sorted(set(names)):
        hit = sorted((_words(n) - cats) & words)
        if hit:
            out.append(f"{hit} of hidden name {n!r} shows in the public manifest")
    shown = {i[k:k + LEAK_WINDOW] for i in ids for k in range(len(i) - LEAK_WINDOW + 1)}
    for hr in hidden["rounds"].values():
        for oid, info in hr["clips"].items():
            for what, value in (("real clip id", str(info.get("clip", ""))[1:]),
                                ("pcm_sha256", str(info.get("pcm_sha256") or ""))):
                if any(value[k:k + LEAK_WINDOW] in shown for k in range(len(value) - LEAK_WINDOW + 1)):
                    out.append(f"a stretch of the {what} of clip {oid} shows in a public id")
    return out


# ------------------------------------------------------------------------------------------------ checks
def _render_dir(renders: Path, clip: Mapping) -> Path:
    return Path(clip["run_dir"]) / "renders" / clip["render"] if clip.get("run_dir") else renders / clip["render"]


def _check_clip(folder: Path, rid: str, rnd: Mapping, clip: Mapping, entry: Mapping | None,
                spec: Mapping | None = None) -> tuple[dict | None, list]:
    errs: list[str] = []
    try:
        rec = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, [f"{rid}: missing render (no readable {folder / 'record.json'})"]
    if rec.get("status") != "ok":
        return None, [f"{rid}: the render failed ({rec.get('status')}): {str(rec.get('error') or '').strip()[-300:]}"]
    gates = rec.get("gates") or {}
    if gates.get("G1-G6") != "pass" or gates.get("G7") != "pass":
        errs.append(f"{rid}: gates {gates} did not pass")
    n = loop_samples(rnd["bpm"])
    if rec.get("n") != n:
        errs.append(f"{rid}: {rec.get('n')} samples, but one pass at {rnd['bpm']} BPM is {n}")
    flac = folder / "render.flac"
    try:
        sr, _ch, frames = audio_info(flac)
    except (OSError, ValueError) as err:
        return None, errs + [f"{rid}: no readable listening copy {flac} ({err})"]
    one_pass = BARS * 240.0 / rnd["bpm"]
    if sr != SR or frames != n or abs(frames / sr - one_pass) > DURATION_TOL * one_pass:
        errs.append(f"{rid}: the FLAC holds {frames} samples at {sr} Hz, not one pass of {n} at {SR}")
    if sha256_file(flac) != rec.get("flac_sha256"):
        errs.append(f"{rid}: {flac} changed since it was rendered (sha256 differs from its record)")
    try:
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if man.get("audio_sha256") != rec.get("pcm_sha256") or man.get("n") != rec.get("n"):
            errs.append(f"{rid}: the render manifest disagrees with its record")
    except (OSError, json.JSONDecodeError):
        errs.append(f"{rid}: no readable render manifest")
    expect = clip.get("expect")
    if expect and (rec.get("clip") != expect["clip"] or rec.get("pcm_sha256") != expect["pcm_sha256"]
                   or rec.get("flac_sha256") != expect["flac_sha256"]):
        errs.append(f"{rid}: a repeat must show its first showing's render, and this one differs from it")
    want = (entry or {}).get("expect_pcm_sha256")
    if want and rec.get("pcm_sha256") != want:
        errs.append(f"{rid}: made again, but its pcm_sha256 differs from the first showing's")
    level = (spec or {}).get("level")
    got = rec.get("level") or {}
    if level and "buses" in level:          # Day 2 on (CD-53): one measured fader per bus the round's thing moves
        if (got.get("case") != level["case"] or sorted(got.get("buses") or {}) != sorted(level["buses"])
                or got.get("ok") is not True):
            errs.append(f"{rid}: a matched clip plays its own fader on {list(level['buses'])} (CD-53), but its record "
                        f"carries {'no level' if not got else 'the level for ' + str(list(got.get('buses') or {}))} "
                        f"on case {got.get('case')} (ok {got.get('ok')}) for {dict(level)}")
    elif level and ({k: got.get(k) for k in level} != dict(level) or got.get("ok") is not True):
        errs.append(f"{rid}: a clip of an instrument stage plays its own fader (CD-52), but its record carries "
                    f"{'no level' if not got else 'the level ' + str({k: got.get(k) for k in (*level, 'ok')})} for "
                    f"{dict(level)}")
    return rec, errs


def _hidden_clip(rec: Mapping, rnd: Mapping, clip: Mapping) -> dict:
    bus, buses = rec["bus"], rec.get("buses") or {}
    info = {"clip": rec["clip"], "option": clip["option"], "what": clip.get("what"), "stage": rnd["stage"],
            "slot": rnd["slot"], "loop": rnd["loop"], "palette": rec["palette"]["label"]}
    level = rec.get("level")
    if level and "buses" in level:                   # Day 2 on (CD-53): every matched bus's own fader and loudness
        lb = level["buses"]
        first = next(b for b in [*bus.split("+"), *MIX_ORDER] if b in lb)   # the stage's bus; the vamps': the keys
        b = buses[first]
        info.update(trim_db=b["trim_db"], fader_db=b["fader_db"], bus_lufs=lb[first]["lufs"], bus_lu=lb[first]["lu"])
        for other in MIX_ORDER:
            if other in lb and other != first:
                info.update({f"{other}_fader_db": buses[other]["fader_db"], f"{other}_lufs": lb[other]["lufs"],
                             f"{other}_lu": lb[other]["lu"]})
    elif bus in buses and level:                     # its own fader, and the loudness measured after it (CD-52)
        b = buses[bus]
        info.update(trim_db=b["trim_db"], fader_db=b["fader_db"], bus_lufs=level["lufs"], bus_lu=level["lu"])
    elif bus in buses:
        b = buses[bus]
        info.update(trim_db=b["trim_db"], fader_db=b["fader_db"], bus_lufs=b["post_lufs"])
    elif bus == "keys+bass":                         # vamps: the harmony moves the keys and the bass (CD-27)
        info.update(trim_db=buses["keys"]["trim_db"], fader_db=buses["keys"]["fader_db"],
                    bus_lufs=buses["keys"]["post_lufs"], bass_lufs=buses["bass"]["post_lufs"])
    else:                                            # rain: the level is the candidate
        info.update(trim_db=None, fader_db=None, bus_lufs=rec.get("bed_lufs"))
    lo_hi = rec.get("sax_range")
    info["sax_range"] = f"{lo_hi[0]}-{lo_hi[1]}" if lo_hi else None
    if clip.get("kit"):
        info["kit"] = clip["kit"]
    info["render"] = rec["render"]
    info["pcm_sha256"] = rec["pcm_sha256"]
    return info


def _draw_id(rng: Rng, prefix: str, n: int, avoid: set[str], used: set[str]) -> str:
    while True:
        x = prefix + rng.hex(n)
        if x not in used and not any(w in x for w in avoid):
            used.add(x)
            return x


# ------------------------------------------------------------------------------------------------ build
def build(doc: Mapping, renders_dir: Path | str, out_root: Path | str, *, summary_dir: Path | str,
          plan: Plan | None = None, check: bool = True) -> list[Path]:
    """Every batch of a draw, from its renders, into `out_root`. Returns the batch folders written."""
    renders_dir = Path(renders_dir).resolve()
    out_root, summary_dir = Path(out_root), Path(summary_dir)
    errors: list[str] = []
    if plan is not None and doc["plan"]["sha256"] != plan.sha256:
        errors.append(f"the draw was made from another plan (sha256 {doc['plan']['sha256'][:12]}...), not "
                      f"{plan.path} ({plan.sha256[:12]}...)")
    seed = int(doc["seed"], 16)
    recs: dict[str, tuple[Path, dict]] = {}          # render folder -> (folder, its record)

    def rec_of(c: Mapping) -> tuple[Path, dict] | None:
        return recs.get(str(_render_dir(renders_dir, c)))

    for b in doc["batches"]:
        if not BATCH_RE.fullmatch(b["name"]):
            errors.append(f"batch name {b['name']!r} must match {BATCH_RE.pattern}")
        target = out_root / b["date"] / b["name"]
        if target.exists():
            errors.append(f"{target} exists; a built batch is never overwritten")
        for r in b["rounds"]:
            for c in r["clips"]:
                folder = _render_dir(renders_dir, c)
                if str(folder) in recs:
                    continue
                entry = doc["renders"].get(c["render"])
                spec = (doc.get("palettes") or {}).get(entry["palette"]) if entry else None
                rec, errs = _check_clip(folder, c["render"], r, c, entry, spec)
                errors += errs
                if rec is not None and not errs:
                    recs[str(folder)] = (folder, rec)
            reals = [rec_of(c)[1]["clip"] for c in r["clips"] if rec_of(c)]
            if len(set(reals)) != len(reals):
                errors.append(f"{b['name']} round {r['slot']}: a real clip appears twice in one round")
    if errors:
        raise BuildError(errors)

    used = {r["of"] for b in doc["batches"] for r in b["rounds"] if r.get("of")}
    built = []
    for b in doc["batches"]:
        hidden_rounds: dict[str, dict] = {}
        draft = []
        for r in b["rounds"]:
            clips = [(c, *rec_of(c)) for c in r["clips"]]
            draft.append((r, clips, {"kind": r["kind"], "brief_id": r["loop"], "category": r["category"],
                                     "clips": {str(i): _hidden_clip(rec, r, c) for i, (c, _, rec) in enumerate(clips)}}))
        avoid = _secret_words({str(i): d[2] for i, d in enumerate(draft)})
        rng = Rng.keyed(seed, "ids", b["name"])
        pub_rounds, ledger = [], []
        for r, clips, hr in draft:
            rid = _draw_id(rng, "r", 8, avoid, used)
            oids = [_draw_id(rng, "k", 10, avoid, used) for _ in clips]
            pub_rounds.append({"id": rid, "bpm": r["bpm"],
                               "clips": [{"id": o, "file": f"{o}.flac", "sha256": rec["flac_sha256"]}
                                         for o, (_, _, rec) in zip(oids, clips)]})
            hidden_rounds[rid] = {"kind": r["kind"], "of": r.get("of"),
                                  "anchor_clip": oids[r["anchor_pos"] - 1] if r["kind"] == "anchor" else None,
                                  "brief_id": r["loop"], "category": r["category"],
                                  "clips": {o: hr["clips"][str(i)] for i, o in enumerate(oids)}}
            ledger.append({"id": rid, "index": len(ledger) + 1, "stage": r["stage"], "slot": r["slot"],
                           "kind": r["kind"], "loop": r["loop"], "category": r["category"], "of": r.get("of"),
                           "anchor_clip": hidden_rounds[rid]["anchor_clip"],
                           "clips": [{"id": o, "pos": i + 1, "option": c["option"], "what": c.get("what"),
                                      "render": rec["render"], "clip": rec["clip"], "pcm_sha256": rec["pcm_sha256"],
                                      "flac_sha256": rec["flac_sha256"],
                                      **({"run_dir": c["run_dir"]} if c.get("run_dir") else {}),
                                      **({"kit": c["kit"]} if c.get("kit") else {})}
                                     for i, (o, (c, _, rec)) in enumerate(zip(oids, clips))]})
        public = {"v": SCHEMA_VERSION, "batch": b["name"], "mode": "shootout", "scene": "city at night",
                  "note": b["note"], "bars": BARS, "sr": SR, "est_min": b["est_min"], "rounds": pub_rounds}
        hidden = {"v": SCHEMA_VERSION, "batch": b["name"], "category": b["category"], "rounds": hidden_rounds}
        leaks = public_leaks(public, hidden, _stage_options(plan, {r["stage"] for r in b["rounds"]}))
        if leaks:
            errors += [f"{b['name']}: {x}" for x in leaks]
        files = [(src / "render.flac", pc["file"]) for pr, (r, clips, _) in zip(pub_rounds, draft)
                 for pc, (_, src, _) in zip(pr["clips"], clips)]
        built.append((b, public, hidden, files, ledger))
    if errors:
        raise BuildError(errors)

    out_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=out_root))
    try:
        folders = []
        for b, public, hidden, files, _ in built:
            d = staging / b["date"] / b["name"]
            d.mkdir(parents=True)
            for src, name in files:
                shutil.copyfile(src, d / name)
            (d / HIDDEN_NAME).write_text(json.dumps(hidden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (d / PUBLIC_NAME).write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            folders.append(d)
        checks = check_folders(folders) if check else []
        bad = [f"{d.name}: {out}" for d, (rc, out) in zip(folders, checks) if rc != 0]
        if bad:
            raise BuildError(["ui.manifest check failed: " + x for x in bad])
        moved = []
        for b, *_ in built:
            target = out_root / b["date"] / b["name"]
            if target.exists():
                raise BuildError([f"{target} appeared while building"])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / b["date"] / b["name"], target)
            moved.append(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ledger_doc = {"schema": BATCHES_SCHEMA, "v": 1, "run": renders_dir.parent.name, "stage": doc["stage"],
                  "seed": doc["seed"],
                  "batches": {b["name"]: {"date": b["date"], "dir": f"{b['date']}/{b['name']}",
                                          "category": b["category"], "est_min": b["est_min"], "rounds": ledger}
                              for b, _, _, _, ledger in built}}
    summary_dir.mkdir(parents=True, exist_ok=True)
    tmp = summary_dir / ".batches.json.tmp"
    tmp.write_text(json.dumps(ledger_doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(summary_dir / "batches.json")
    return moved


def _stage_options(plan: Plan | None, stages: set[str]) -> set[str]:
    """Every candidate of the lists the stages draw from (a note may name none of them)."""
    out: set[str] = set()
    for sid in stages:
        st = plan.stages.get(sid) if plan is not None else None
        for g in st.groups if st else ():
            out |= set(plan.band_options[g.leaders] if g.leaders else g.options)
    return out


def check_folders(folders: list[Path]) -> list[tuple[int, str]]:
    """`python -m lofi_rl.ui.manifest check <folder>` for each folder, each in a fresh interpreter as the rating
    page's server would load it, all at once. (exit code, output) per folder, in order."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    procs = [subprocess.Popen([sys.executable, "-m", "lofi_rl.ui.manifest", "check", str(d)], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env) for d in folders]
    out = []
    for p in procs:
        text, _ = p.communicate()
        out.append((p.returncode, text.strip()))
    return out


# ------------------------------------------------------------------------------------------------ CLI
def _seed(text: str | None) -> int:
    if text is None:
        return _secrets.randbits(64)
    try:
        value = int(text, 16)
    except ValueError:
        raise DrawError(f"--seed {text!r} is not hexadecimal") from None
    if not 0 <= value < 1 << 64:
        raise DrawError("--seed must fit in 64 bits")
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.shootout.batch", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="draw the day's maps, order and hidden controls into summary/draw.json")
    d.add_argument("--plan", type=Path, default=PLAN)
    d.add_argument("--stage", required=True, choices=("day1", "day2", "day3"))
    d.add_argument("--summary", required=True, type=Path)
    d.add_argument("--leaders", default=None)
    d.add_argument("--prior", type=Path, nargs="*", default=[], help="earlier runs' summary/batches.json")
    d.add_argument("--repeats", default="", help="stages whose rounds may come back, joined by +")
    d.add_argument("--rerun-s2", default=None)
    d.add_argument("--cut", default="", help="cut-list names joined by +")
    d.add_argument("--seed", default=None, help="16 hex digits (tests only; the job draws a fresh seed)")
    b = sub.add_parser("build", help="build the stage's batches from the draw and the renders")
    b.add_argument("--plan", type=Path, default=PLAN)
    b.add_argument("--stage", required=True, choices=("day1", "day2", "day3"))
    b.add_argument("--renders", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path, help="the inbox root")
    b.add_argument("--draw", type=Path, default=None, help="default: <renders>/../summary/draw.json")
    b.add_argument("--summary", type=Path, default=None, help="where batches.json goes (default: the draw's folder)")
    a = p.parse_args(argv)
    try:
        plan = load_plan(a.plan)
        if a.cmd == "draw":
            doc = draw(plan, a.stage, seed=_seed(a.seed), leaders=parse_leaders(a.leaders),
                       priors=[load_prior(x) for x in a.prior], repeats=[x for x in a.repeats.split("+") if x],
                       rerun_s2=a.rerun_s2, cuts=[x for x in a.cut.split("+") if x])
            path = write_draw(doc, a.summary / "draw.json")
            n = sum(len(x["rounds"]) for x in doc["batches"])
            print(f"draw: {path}: {len(doc['batches'])} batches, {n} rounds, {len(doc['renders'])} renders, "
                  f"{len(doc['palettes'])} palettes, {len(doc['sour'])} sour copies (seed recorded there only)")
            return 0
        draw_path = a.draw or a.renders.parent / "summary" / "draw.json"
        doc = json.loads(draw_path.read_text(encoding="utf-8"))
        if doc.get("stage") != a.stage:
            print(f"error: {draw_path} is the draw of {doc.get('stage')}, not {a.stage}", file=sys.stderr)
            return 2
        folders = build(doc, a.renders, a.out, summary_dir=a.summary or draw_path.parent, plan=plan)
    except (PlanError, DrawError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 3
    for f, x in zip(folders, doc["batches"]):
        clips = sum(len(r["clips"]) for r in x["rounds"])
        print(f"batch {x['name']}: {len(x['rounds'])} rounds, {clips} clips, est {x['est_min']} min -> {f}")
    print(f"build: {len(folders)} batches checked by ui.manifest; ids in "
          f"{(a.summary or draw_path.parent) / 'batches.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
