"""render: the held-out renders and the sets (docs/design/step0_samples.md §2.5, §3), in the scorer env.

    python -m lofi_rl.step0 render --run DIR [--workers 4] [--deadline-s 900] [--cache DIR]
    python -m lofi_rl.step0 render --from OUT --run REDO [...]      a recovery (§4.2): OUT read-only, REDO written

- **Only a complete run is rendered.** A partial one (generate stopped or unfinished, fewer rows than `requested`,
  score's `partial`, an unreadable line, a scored row that is not its completion's) is refused with exit 2 and nothing
  written, since its sets could not count.
- **The render-code key**: render_code = {stamp, source_sha8}, this step's .sync-stamp and
  lofi_rl.scoring.features.source_sha8() (the render and feature code), in clips.json, sets.json and step0.json, so
  every G7 outcome is known with the code that rendered it (§3).
- **Rounds** (select.pending, the lazy rule of §1.3-1.4). Round 1 renders each held-out brief's first 4 distinct
  candidates; each later round renders, for each brief still short of 4, its next distinct candidates, and once more a
  draw whose render failed by infrastructure the round before. Within a round the draws go in stream order, then k,
  through junk.render_clip, two at a time with --workers split between them (junk.render_lanes, junk.in_order): the
  lanes never change a clip, and the records keep the round's order (D-022).
- **Outcomes.** A G7 failure is final and the draw is not taken. An infrastructure error (never a gate: junk.render_clip
  catches it, and an exception it lets out, writing the FLAC say, counts the same) is retried once in the next round;
  a second one is final and leaves the brief undetermined. No render starts after --deadline-s, counted from the
  step's start; a skipped render is final and leaves its brief undetermined.
- **Files**: clips/<i:04d>.flac (one pass, 16-bit) and its .manifest.json, as junk.render_clip writes them. Each record
  is junk.render_clip's plus brief_id, k, round, final, clip_id and render_cfg_sha8. The clip id (taste_pipeline.md
  §2) is the one lofi_rl.shootout.render.clip_id gives the manifest, from the manifest's canonical sha256, brief,
  palette and render config, through lofi_rl.scoring.protocol.clip_id; render_cfg_sha8 is that config's, or for a
  record with no render the render pool's (render_pool.render_config), so the laptop can check every clip id from
  recorded inputs.
- **At the end**: sets.json (select.sets_json over scored.jsonl and the records: the final held-out set, then Ref_b and
  the pool's 220 with their picks by their records, the pool skipping the final held-out set's texts under Q5(a)),
  clips.json, and step0.json's render, render_code, sets and render_stamp. Ref_b and the pool are not rendered here.

`--from OUT --run REDO` reads generate's records (generate.json, completions.jsonl) from OUT, and score's (scored.jsonl,
step0.json) from REDO when a score recovery wrote them there, else from OUT, copying scored.jsonl into REDO unchanged,
so REDO holds score's and render's records together (check reads both from REDO). It never writes under OUT.

Exit 0: the held-out set is decided. 1: a held-out brief is undetermined, or a render failed a gate other than G7 (a
draw that passed score's G1-G6 and compile but not the render's: a defect between the two). 2: bad input, a partial
run, or --run and --from that overlap, with nothing rendered or written. The audio code (junk.render_clip's, the
render pool's config, source_sha8) is imported inside functions, since the policy env's build imports every lofi_rl
module outside its scorer-side list (jobs/env_policy.sbatch). No claim about sound: clips are only measured (length,
loudness, hashes).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl.brief import Brief
from lofi_rl.brief_streams import ROOT
from lofi_rl.palette import Palette, PaletteError, load_palette
from lofi_rl.rl import junk as J
from lofi_rl.scoring import protocol
from lofi_rl.step0 import Step0Error
from lofi_rl.step0 import requests as R
from lofi_rl.step0 import select as X
from lofi_rl.step0.score import read_json, separate, sync_stamp, write_json

SKIPPED = "render deadline"             # a skipped record's reason, as junk's render step writes it
READ = ("generate.json", "completions.jsonl", "scored.jsonl", "step0.json")


def _now() -> float:
    return time.monotonic()


def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------------------------------------------- the render-code key
def source_sha8() -> str:
    """lofi_rl.scoring.features.source_sha8(): the render and feature code (features imports numpy: scorer env)."""
    from lofi_rl.scoring.features import source_sha8 as sha8
    return sha8()


def pool_cfg_sha8() -> str:
    """The render config's sha8 (taste_pipeline.md §2) as the render pool builds the config."""
    from lofi_rl.scoring.render_pool import render_config
    return protocol.render_cfg_sha8(render_config())


def clip_of(manifest: Mapping) -> tuple[str, str]:
    """(clip id, render_cfg_sha8) of a render from its manifest: the id lofi_rl.shootout.render.clip_id gives it
    (the canonical loop's sha256, the brief's bpm, key, mode and energy, the palette's id and sha8 and the render config
    {schema, chain, flac}), through lofi_rl.scoring.protocol.clip_id."""
    cfg = {"schema": manifest["schema"], "chain": manifest["chain"], "flac": protocol.FLAC_FORMAT}
    b, p = manifest["brief"], manifest["palette"]
    brief = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"], scene=b["scene"])
    return (protocol.clip_id(manifest["loop"]["canon_sha256"], brief, p["id"], p["sha256"], cfg),
            protocol.render_cfg_sha8(cfg))


# --------------------------------------------------------------------------------------------- the run's records
def partial(gen: dict, step0: dict, rows: list[dict], bad_rows: int, scored: list[dict], bad_scored: int) -> list[str]:
    """Why a run is not complete by its own records; empty when it is (module doc)."""
    out = []
    requested = gen.get("requested")
    if gen.get("schema") != S.GENERATE_SCHEMA:
        out.append(f"generate.json is not {S.GENERATE_SCHEMA}")
    if gen.get("stopped") is not None:
        out.append(f"generate stopped: {str(gen['stopped'])[:300]}")
    if gen.get("partial") is not False:
        out.append("generate.json is partial: generate did not finish")
    if not _whole(requested) or gen.get("generated") != requested:
        out.append(f"generate drew {gen.get('generated')} of {requested} requests")
    if step0.get("schema") != S.STEP0_SCHEMA:
        out.append(f"step0.json is not {S.STEP0_SCHEMA}")
    if step0.get("partial") is not False:
        out.append("score marked the run partial")
    if bad_rows or bad_scored or step0.get("unreadable_lines"):
        out.append(f"unreadable lines: completions.jsonl {bad_rows}, scored.jsonl {bad_scored}, step0.json "
                   f"{step0.get('unreadable_lines')}")
    if (len(rows), len(scored), step0.get("n")) != (requested, requested, requested):
        out.append(f"{len(rows)} completions and {len(scored)} scored rows (step0.json n {step0.get('n')}) for "
                   f"{requested} requests")
    by_i = {r["i"]: r for r in rows}
    if len(by_i) != len(rows) or len({s["i"] for s in scored}) != len(scored):
        out.append("a row index repeats in completions.jsonl or scored.jsonl")
    keys = ("stream", "brief_id", "k")
    odd = [s["i"] for s in scored if [(by_i.get(s["i"]) or {}).get(k) for k in keys] != [s.get(k) for k in keys]]
    if odd:
        out.append(f"{len(odd)} scored rows are not their completion's draw (first i={odd[0]})")
    return out


def sources(run: Path, out: Path | None) -> tuple[Path, Path]:
    """(the folder of generate's records, the folder of score's): RUN's own; with --from OUT, OUT's and REDO's when a
    score recovery wrote scored.jsonl there, else OUT's."""
    if out is None:
        return run / "summary", run / "summary"
    redo = run / "summary"
    return out / "summary", (redo if (redo / "scored.jsonl").is_file() else out / "summary")


# --------------------------------------------------------------------------------------------- the renders
def attempt(row: dict, palette: Palette, clips_dir: Path, *, cache: Path | None, workers: int) -> tuple[dict, str | None,
                                                                                                     str | None]:
    """One render through junk.render_clip, with its clip id and render_cfg_sha8 from the manifest it wrote (None for a
    record with no render). An exception render_clip lets out is an infrastructure error, as one inside it is."""
    try:
        rec = J.render_clip(row, palette, clips_dir, cache=cache, workers=workers)
        if rec.get("ok") is True:
            manifest = json.loads((clips_dir / f"{row['i']:04d}.manifest.json").read_text(encoding="utf-8"))
            return rec, *clip_of(manifest)
    except Exception as err:  # noqa: BLE001: files, FluidSynth: infrastructure, never a gate
        rec = {"i": row["i"], "ok": False, "gate": None, "error": f"{type(err).__name__}: {err}"[:500]}
    return rec, None, None


def rounds(rows: Mapping[int, dict], scored: list[dict], streams, palette: Palette, clips_dir: Path, *, jobs: int,
           each: int, cache: Path | None, deadline_s: float, t0: float, on_record) -> list[dict]:
    """Every round of held-out renders (module doc), until select.pending names none; `on_record` sees each record as
    it is kept, in the round's order."""
    cfg8 = pool_cfg_sha8()
    records: list[dict] = []
    rnd = 0
    while True:
        todo = X.pending(scored, records, streams=streams)
        if not todo:
            return records
        rnd += 1
        tried = frozenset(r["i"] for r in records)  # a pending draw with a record failed once by infrastructure

        def one(i: int, rnd: int = rnd, tried: frozenset = tried) -> dict:
            row = rows[i]
            if _now() - t0 >= deadline_s:           # read as the render would start: none starts after the deadline
                rec, clip, sha8 = {"i": i, "ok": False, "gate": None, "skipped": SKIPPED}, None, None
            else:
                rec, clip, sha8 = attempt(row, palette, clips_dir, cache=cache, workers=each)
            final = not (rec.get("error") and i not in tried)
            return {**rec, "brief_id": row["brief_id"], "k": row["k"], "round": rnd, "final": final, "clip_id": clip,
                    "render_cfg_sha8": sha8 or cfg8}
        for rec in J.in_order(one, todo, jobs):
            records.append(rec)
            on_record(rec)


def _say(rec: dict) -> str:
    return "ok" if rec["ok"] else rec.get("gate") or rec.get("skipped") or (rec.get("error") or "")[:120]


def main(a: argparse.Namespace) -> int:
    """The `render` subcommand (lofi_rl.step0.cli)."""
    t0 = _now()
    run = Path(a.run)
    out = None if a.from_ is None else Path(a.from_)
    problems = [] if out is None else separate(out, run)
    gen_dir, score_dir = sources(run, out)
    files = {name: (gen_dir if name in READ[:2] else score_dir) / name for name in READ}
    problems += [f"{p} is missing" for p in files.values() if not p.is_file()]
    if problems:
        print("render: nothing rendered or written:\n" + "\n".join(f"- {p}" for p in problems), file=sys.stderr)
        return 2
    gen, step0 = read_json(files["generate.json"]) or {}, read_json(files["step0.json"]) or {}
    rows, bad_rows = J.read_jsonl(files["completions.jsonl"])
    scored, bad_scored = J.read_jsonl(files["scored.jsonl"])
    problems = partial(gen, step0, rows, bad_rows, scored, bad_scored)
    if problems:
        print("render: a partial run is not rendered (its sets could not count):\n"
              + "\n".join(f"- {p}" for p in problems), file=sys.stderr)
        return 2
    try:
        streams = R.load()
        X.pending(scored, [], streams=streams)              # the rows are one run's draws: select refuses others
        palette = load_palette(ROOT / S.PALETTE, verify=True)
    except (Step0Error, PaletteError, OSError) as err:
        print(f"render: nothing rendered or written:\n{err}", file=sys.stderr)
        return 2
    render_code = {"stamp": sync_stamp(), "source_sha8": source_sha8()}
    summ, clips_dir = run / "summary", run / "clips"
    summ.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    if score_dir != summ:                                   # a render recovery: score's records, unchanged, in REDO
        shutil.copyfile(files["scored.jsonl"], summ / "scored.jsonl")
    jobs, each = J.render_lanes(J.RENDER_JOBS, a.workers)
    clips = {"schema": S.CLIPS_SCHEMA, "render_code": render_code,
             "palette": {"id": palette.id, "sha256": palette.sha256}, "jobs": jobs, "workers": each, "clips": []}

    def keep(rec: dict) -> None:
        clips["clips"].append(rec)
        write_json(summ / "clips.json", clips)
        print(f"[step0:render] round {rec['round']} {len(clips['clips'])} i={rec['i']} {rec['brief_id']} k{rec['k']} "
              f"{_say(rec)}", flush=True)
    records = rounds({r["i"]: r for r in rows}, scored, streams, palette, clips_dir, jobs=jobs, each=each,
                     cache=a.cache, deadline_s=a.deadline_s, t0=t0, on_record=keep)
    write_json(summ / "clips.json", clips)
    sets = X.sets_json(scored, records, streams=streams, render_code=render_code)
    write_json(summ / "sets.json", sets)
    blk = X.render_block(records)
    step0.update(render=blk, render_code=render_code, sets=X.readouts(sets, scored), render_stamp=sync_stamp())
    write_json(summ / "step0.json", step0)
    held = sets["heldout"]
    undetermined = [e["brief_id"] for e in held if e["undetermined"]]
    print(f"RENDER rendered {blk['rendered']} of {blk['attempted']} attempted in {blk['rounds']} rounds: G7 {blk['G7']}, "
          f"other gates {blk['other_gates']}, errors retried {blk['errors_retried']} and final {blk['errors_final']}, "
          f"skipped {blk['skipped']} | held-out {sum(len(e['taken']) for e in held)} clips in {len(held)} briefs, "
          f"{sum(e['short'] for e in held)} short, {len(undetermined)} undetermined | Ref_b "
          f"{sum(len(e['taken']) for e in sets['refb'])} by its records | pool {sum(len(e['taken']) for e in sets['pool'])}"
          f" picks in {len(sets['pool'])} briefs, {sum(e['short'] for e in sets['pool'])} short", flush=True)
    if undetermined:
        print(f"render: {len(undetermined)} held-out briefs are undetermined (first {undetermined[0]}): the held-out set "
              "is not decided (§4.2)", file=sys.stderr)
    return 1 if undetermined or blk["other_gates"] else 0
