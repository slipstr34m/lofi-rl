"""Scorer: the audio part of the reward for one batch (docs/SCORER.md §2, §5; taste_pipeline.md §4-5).

For each request:
1. **Parse** every text with the palette's vocabulary in the brief's mode; its canonical form gives the clip id
   (protocol.clip_id). Texts with one canonical form, brief and palette are one clip: rendered and scored once.
2. **Records**: each distinct clip from the in-memory LRU, else the disk cache, else rendered (render pool) and
   featurized (MERT, CLAP on the mix and the music stem, Audiobox), one clip on the GPU while others render.
3. **Score** each item from its clip's record:

       hard     = parse and compile and render in spec (the record's ok)
       G        = hard x 1[PQ >= tau_PQ] x clip(P_acc / 0.5, 0.25, 1)                         (§5 "Gate")
       r_theme  = sigma((m - med) / (1.4826 MAD)),  m the CLAP margin of the music stem         (theme.py)
       r_ce     = sigma((CE - med) / (1.4826 MAD)) when CE is validated, else 0                 (calib.py)
       r_taste  = sigma(mean_m d_m - std_m d_m), d_m = u_m - mean of the brief's 4 references (rm_api.par)

   A missing calibration disables its term or factor (tau_PQ: factor 1; theme: 0; CE: 0; RM: r_taste 0 and
   P_acc factor 1; references: r_taste 0), and every item's diag.disabled names what is off.

The backends are injected (render pool, featurizer, theme, RM, calibration, cache), so the laptop tests run this
class and the HTTP server with fakes (lofi_rl.scoring.fakes). Requests run one at a time (`lock`): the models are
not thread-safe and the pool is shared.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lofi_rl.dsl import canon, canon_sha, parse
from lofi_rl.palette import Palette

from . import protocol
from .cache import ClipRecord, FeatureCache
from .calib import Calibration, robust_sigmoid
from .protocol import AUDIO_TERMS, DIAG_KEYS, PROTOCOL, SIG_KEYS, Item, ProtocolError, RenderRequest, ScoreRequest
from .render_pool import RenderJob, RenderPoolDrift, drift
from .rm_api import RMState, par
from .theme import Theme

P_ACC_HALF, P_ACC_FLOOR = 0.5, 0.25           # clip(P_acc / 0.5, 0.25, 1)


class RequestError(Exception):
    status = 400


class Conflict(RequestError):
    """The request was made for another palette or another pinned version."""
    status = 409


class Unprocessable(RequestError):
    """Valid JSON the server cannot score: a brief without PAR references, an unusable out_dir."""
    status = 422


def gate_factor(hard: int, pq: float | None, tau_pq: float | None, p_acc: float | None) -> tuple[float, float, float]:
    """(G, PQ factor, P_acc factor) of §5's gate. tau_pq None: no PQ floor; p_acc None: no RM."""
    pq_f = 1.0 if tau_pq is None else (1.0 if pq is not None and pq >= tau_pq else 0.0)
    pacc_f = 1.0 if p_acc is None else min(1.0, max(P_ACC_FLOOR, p_acc / P_ACC_HALF))
    return (float(hard) * pq_f * pacc_f if hard else 0.0), pq_f, pacc_f


@dataclass(frozen=True)
class Prepared:
    item: Item
    clip_id: str
    canon: str | None               # None: the text did not parse
    gate: str | None = None         # the parser's gate code when it did not
    detail: str | None = None


class _Timer:
    """Count, sum and max of one stage's milliseconds."""

    def __init__(self) -> None:
        self.n, self.sum, self.max = 0, 0.0, 0.0

    def add(self, ms: float | None) -> None:
        if ms is None:
            return
        self.n += 1
        self.sum += float(ms)
        self.max = max(self.max, float(ms))

    def doc(self) -> dict[str, float]:
        return {"n": self.n, "sum": round(self.sum, 1), "mean": round(self.sum / self.n, 2) if self.n else None,
                "max": round(self.max, 2)}


def _finite_list(xs: Any, n: int, what: str) -> list[float]:
    out = [float(x) for x in xs]
    if len(out) != n or not all(math.isfinite(x) for x in out):
        raise FloatingPointError(f"the RM returned {what} {out!r} for {n} members")
    return out


class Scorer:
    def __init__(self, *, palette: Palette, renderer: Any, featurizer: Any, theme: Theme, calib: Calibration,
                 rm: RMState, cache: FeatureCache, render_cfg: Mapping[str, Any], backend: str) -> None:
        self.palette, self.vocab = palette, palette.vocab()
        self.renderer, self.featurizer, self.theme = renderer, featurizer, theme
        self.calib, self.rm, self.cache, self.backend = calib, rm, cache, backend
        self.render_cfg = dict(render_cfg)
        self.versions = {"reward_ver": protocol.REWARD_VER, "feat_ver": featurizer.feat_ver, "rm_ver": rm.rm_ver,
                         "render_ver": protocol.render_version(render_cfg)}
        if cache.feat_ver != featurizer.feat_ver:
            raise ValueError(f"the cache is for {cache.feat_ver}, the featurizer makes {featurizer.feat_ver}")
        if cache.hw is not None and cache.hw != getattr(featurizer, "hw", None):
            raise ValueError(f"the cache is for hardware {cache.hw!r}, the featurizer runs on "
                             f"{getattr(featurizer, 'hw', None)!r}")
        made_for = getattr(featurizer, "render", None)
        if made_for is not None and made_for != renderer.name:
            raise ValueError(f"the featurizer's feat_ver names the {made_for!r} renderer, but this scorer renders "
                             f"with {renderer.name!r}: its records would be cached as the other renderer's")
        self.lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.t0 = time.time()
        self.counts: Counter[str] = Counter()
        self.timers = {k: _Timer() for k in ("request", "parse", "render", "featurize", "mert", "clap", "audiobox",
                                              "score")}

    # -------------------------------------------------------------------------------------- state
    def terms(self) -> dict[str, bool]:
        c, rm = self.calib, self.rm
        return {"pq_gate": c.pq is not None, "pacc_gate": rm.pacc_live, "ce": c.ce_validated,
                "theme": c.theme is not None, "taste": rm.taste_live}

    def disabled(self) -> list[str]:
        return [k for k, v in self.terms().items() if not v]

    def health(self) -> dict[str, Any]:
        cfg = self.theme.config
        return {"ok": True, "protocol": PROTOCOL, "backend": self.backend, "versions": dict(self.versions),
                "terms": self.terms(), "disabled": self.disabled(), "palette_id": self.palette.id,
                "palette_sha256": self.palette.sha256, "palette_path": self.palette.path,
                "calibration": {**self.calib.flags(), "rm": self.rm.flags(),
                                "theme_config": {"path": cfg.path, "sha256": cfg.sha256, "version": cfg.version}},
                "renderer": {"name": self.renderer.name, "workers": getattr(self.renderer, "workers", None),
                             "timeout_s": getattr(self.renderer, "timeout_s", None)},
                "featurizer": self.featurizer.info(), "gpu": self.featurizer.gpu(), "cache": self.cache.counters(),
                "pid": os.getpid(), "uptime_s": round(time.time() - self.t0, 1)}

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            counts = dict(self.counts)
            timers = {k: t.doc() for k, t in self.timers.items()}
        return {"protocol": PROTOCOL, "versions": dict(self.versions), "uptime_s": round(time.time() - self.t0, 1),
                "counts": counts, "t_ms": timers, "cache": self.cache.counters(), "gpu": self.featurizer.gpu(),
                "pool": {"workers": getattr(self.renderer, "workers", None),
                         "rebuilds": getattr(self.renderer, "rebuilds", 0)}}

    def _count(self, **kw: int) -> None:
        with self._stats_lock:
            self.counts.update(kw)

    def _time(self, stage: str, ms: float | None) -> None:
        with self._stats_lock:
            self.timers[stage].add(ms)

    # -------------------------------------------------------------------------------------- checks
    def _check_palette(self, sha: str) -> None:
        if sha != self.palette.sha256:
            raise Conflict(f"the request was made for palette sha256 {sha[:12]}..., but this server renders "
                           f"{self.palette.id} ({self.palette.sha256[:12]}...)")

    def _check_expect(self, expect: Mapping[str, str]) -> None:
        bad = {k: (v, self.versions[k]) for k, v in expect.items() if self.versions.get(k) != v}
        if bad:
            raise Conflict("pinned versions differ from this server's: " +
                           ", ".join(f"{k} {a!r} != {b!r}" for k, (a, b) in sorted(bad.items())))

    def _check_refs(self, items: Sequence[Item]) -> None:
        if not self.rm.taste_live:
            return
        refs = self.rm.refs.refs
        missing = sorted({str(it.brief_id) for it in items if it.brief_id not in refs})
        if missing:
            raise Unprocessable(f"taste is live, but {self.rm.refs.path} has no step-0 references for brief_id "
                                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}: PAR cannot centre their utilities")

    def _check_render(self, out: Mapping[str, Any], clip_id: str) -> None:
        """Refuse a render from a worker that runs other code or another palette than feat_ver and the clip ids
        name, or whose manifest gives another clip id: RenderPoolDrift, an HTTP 500, so the trainer crashes and
        the chain restarts a coherent server (a per-item render-error would hide the drift). Nothing is cached
        from such a render."""
        w = out.get("worker")
        if w is not None:
            want = {"src_sha8": getattr(self.featurizer, "src_sha8", None), "palette_sha256": self.palette.sha256}
            bad = drift(w, want, f"render worker {w.get('pid')}")
            if bad:
                raise RenderPoolDrift(bad)
        man = out.get("manifest")
        if man is not None:
            from lofi_rl.shootout.render import clip_id as manifest_clip_id
            got = manifest_clip_id(man)
            if got != clip_id:
                raise RenderPoolDrift(f"the render of {clip_id} carries the manifest of clip {got}: the render "
                                      f"workers and the server disagree on the clip id")

    # -------------------------------------------------------------------------------------- stages
    def prepare(self, items: Sequence[Item]) -> list[Prepared]:
        out = []
        pid, psha = self.palette.id, self.palette.sha256
        for it in items:
            t = time.perf_counter()
            gate = detail = None
            try:
                res = parse(it.text, vocab=self.vocab, mode=it.brief.mode)
                ok = res.gate.ok and res.ast is not None
                if not ok:
                    gate, detail = res.gate.code or "G1", res.gate.detail
            except Exception as err:  # noqa: BLE001: a parser crash is this item's failure, never the batch's
                res, ok = None, False
                gate, detail = "parse-error", f"{type(err).__name__}: {err}"
            if ok:
                cid = protocol.clip_id(canon_sha(res.ast), it.brief, pid, psha, self.render_cfg)
                out.append(Prepared(it, cid, canon(res.ast)))
            else:
                out.append(Prepared(it, protocol.raw_clip_id(it.text, it.brief, pid, psha, self.render_cfg),
                                    None, gate, (detail or "")[:500]))
            self._time("parse", 1000 * (time.perf_counter() - t))
        return out

    def _records(self, unique: Mapping[str, Prepared]) -> dict[str, tuple[ClipRecord, str]]:
        """Each distinct clip's record and where it came from: "mem", "disk", "miss" (rendered now) or "none"
        (the text did not parse; nothing to render)."""
        got: dict[str, tuple[ClipRecord, str]] = {}
        todo: list[Prepared] = []
        for cid, p in unique.items():
            if p.canon is None:
                got[cid] = (ClipRecord(cid, None, False, p.gate, p.detail), "none")
                continue
            hit = self.cache.get(cid)
            if hit is not None:
                got[cid] = hit
            else:
                todo.append(p)

        def on_done(k: int, out: Mapping[str, Any]) -> None:
            p = todo[k]
            self._check_render(out, p.clip_id)
            self._time("render", out.get("render_ms"))
            if out["ok"]:
                t = time.perf_counter()
                feats = self.featurizer.featurize(p.clip_id, p.canon, out["mix"], out["music"])
                self._time("featurize", 1000 * (time.perf_counter() - t))
                for stage in ("mert", "clap", "audiobox"):
                    self._time(stage, feats.t_ms.get(stage))
                rec = ClipRecord(p.clip_id, p.canon, True, None, None, False, dict(out["info"]), feats)
                self._count(rendered=1)
            else:
                rec = ClipRecord(p.clip_id, p.canon, False, out["gate"], out["error"], bool(out["transient"]),
                                 {"render_ms": out.get("render_ms")})
                self._count(**{f"render_fail_{out['gate']}": 1})
            self.cache.put(rec)
            got[p.clip_id] = (rec, "miss")

        self.renderer.render_many([RenderJob(p.clip_id, p.canon, protocol.brief_dict(p.item.brief)) for p in todo],
                                  on_done)
        missing = [p.clip_id for p in todo if p.clip_id not in got]
        if missing:
            raise RuntimeError(f"the renderer returned nothing for {missing}")
        return got

    def score_item(self, p: Prepared, rec: ClipRecord, kind: str, dup_of: int | None) -> dict[str, Any]:
        t = time.perf_counter()
        disabled = self.disabled()
        diag: dict[str, Any] = {k: None for k in DIAG_KEYS}
        diag.update(disabled=disabled, cache=kind, dup_of=dup_of, error=rec.error, render=rec.render or None,
                    hw=None if rec.feats is None else {"machine": rec.feats.machine, "gpu": rec.feats.gpu})
        sig = {k: None for k in SIG_KEYS}
        r = {k: 0.0 for k in AUDIO_TERMS}
        if not rec.ok or rec.feats is None:
            item = {"i": p.item.i, "clip_id": p.clip_id, "hard": 0, "G": 0.0, "gate": rec.gate or "render-error",
                    "r": r, "sig": sig, "diag": diag}
        else:
            f, c = rec.feats, self.calib
            ab = f.audiobox
            th = self.theme.margin(f.clap_music, p.item.brief.scene)
            m = th["m"]
            r["theme"] = Theme.r_theme(m, c.theme)
            r["ce"] = robust_sigmoid(ab["CE"], c.ce.med, c.ce.mad) if c.ce_validated else 0.0
            p_acc = u = None
            if self.rm.model is not None:
                u = _finite_list(self.rm.model.utilities(f), int(self.rm.model.members), "utilities")
                p_acc = float(self.rm.model.p_acc(f))
                if not (math.isfinite(p_acc) and 0.0 <= p_acc <= 1.0):
                    raise FloatingPointError(f"the RM returned P_acc {p_acc!r}")
                if self.rm.refs is not None:
                    r["taste"], diag["rm_mean"], diag["rm_std"] = par(u, self.rm.refs.means(p.item.brief_id))
            G, pq_f, pacc_f = gate_factor(1, ab["PQ"], None if c.pq is None else c.pq.tau_pq, p_acc)
            sig.update(pq=ab["PQ"], pc=ab["PC"], ce=ab["CE"], cu=ab["CU"], clap_margin=m,
                       centroid_hz=rec.render.get("centroid_hz"))
            diag.update(p_acc=p_acc, pq=ab["PQ"], pc=ab["PC"], cu=ab["CU"], pq_factor=pq_f, pacc_factor=pacc_f,
                        tau_pq=None if c.pq is None else c.pq.tau_pq, u=u,
                        theme={k: th[k] for k in ("pos_mean", "contrast_max", "nearest_contrast", "contrast")})
            item = {"i": p.item.i, "clip_id": p.clip_id, "hard": 1, "G": G, "gate": None, "r": r, "sig": sig,
                    "diag": diag}
        ms = 1000 * (time.perf_counter() - t)
        self._time("score", ms)
        item["t_ms"] = {"render": (rec.render or {}).get("render_ms") if kind == "miss" else 0.0,
                        "features": (rec.feats.t_ms if (rec.feats is not None and kind == "miss") else {}),
                        "score": round(ms, 3)}
        return item

    @staticmethod
    def _unique(prepared: Sequence[Prepared]) -> tuple[dict[str, Prepared], dict[int, int | None]]:
        unique: dict[str, Prepared] = {}
        first: dict[str, int] = {}
        dup_of: dict[int, int | None] = {}
        for p in prepared:
            if p.clip_id in unique:
                dup_of[p.item.i] = first[p.clip_id]
            else:
                unique[p.clip_id], first[p.clip_id] = p, p.item.i
                dup_of[p.item.i] = None
        return unique, dup_of

    # -------------------------------------------------------------------------------------- endpoints
    def score(self, req: ScoreRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        self._check_palette(req.palette_sha256)
        self._check_expect(req.expect)
        self._check_refs(req.items)
        with self.lock:
            prepared = self.prepare(req.items)
            unique, dup_of = self._unique(prepared)
            t1 = time.perf_counter()
            recs = self._records(unique)
            t2 = time.perf_counter()
            items = []
            scored: dict[tuple, dict[str, Any]] = {}     # one score per (clip, brief_id, scene): PAR keys on brief_id
            for p in prepared:
                rec, kind = recs[p.clip_id]
                key = (p.clip_id, p.item.brief_id, p.item.brief.scene)
                if key in scored:
                    first = scored[key]
                    items.append({**first, "i": p.item.i, "r": dict(first["r"]), "sig": dict(first["sig"]),
                                  "diag": {**first["diag"], "cache": "dup", "dup_of": first["i"]},
                                  "t_ms": {"render": 0.0, "features": {}, "score": 0.0}})
                    continue
                kind = "dup" if dup_of[p.item.i] is not None else kind
                items.append(self.score_item(p, rec, kind, dup_of[p.item.i]))
                scored[key] = items[-1]
        kinds = Counter(it["diag"]["cache"] for it in items)
        fails = Counter(it["gate"] for it in items if it["gate"] is not None)
        total = 1000 * (time.perf_counter() - t0)
        self._time("request", total)
        self._count(requests_score=1, items=len(items), unique=len(unique), dup=kinds.get("dup", 0))
        return {"protocol": PROTOCOL, **self.versions, "terms": self.terms(), "disabled": self.disabled(),
                "step": req.step, "items": items,
                "batch": {"n": len(items), "unique": len(unique), "rendered": kinds.get("miss", 0),
                          "cache": {k: kinds.get(k, 0) for k in protocol.CACHE_KINDS}, "gates": dict(fails),
                          "t_ms": {"parse": round(1000 * (t1 - t0), 2), "records": round(1000 * (t2 - t1), 2),
                                   "total": round(total, 2)}}}

    def render(self, req: RenderRequest) -> dict[str, Any]:
        """Render without features or scores; with out_dir, write <clip_id>.flac (with flac) and
        <clip_id>.manifest.json there. Never cached: the point is the audio."""
        t0 = time.perf_counter()
        self._check_palette(req.palette_sha256)
        out_dir = None if req.out_dir is None else Path(req.out_dir)
        if out_dir is not None and not out_dir.is_dir():
            raise Unprocessable(f"out_dir {out_dir} is not a directory")
        with self.lock:
            prepared = self.prepare(req.items)
            unique, dup_of = self._unique(prepared)
            todo = [p for p in unique.values() if p.canon is not None]
            done: dict[str, dict[str, Any]] = {}

            def on_done(k: int, out: Mapping[str, Any]) -> None:
                p = todo[k]
                self._check_render(out, p.clip_id)
                self._time("render", out.get("render_ms"))
                files: dict[str, str] = {}
                if out["ok"] and out_dir is not None and out.get("mix") is not None:
                    from lofi_rl.render.formats import write_flac
                    from lofi_rl.render.pipeline import manifest_json
                    man = out_dir / f"{p.clip_id}.manifest.json"
                    man.write_text(manifest_json(out["manifest"]), encoding="utf-8")
                    files["manifest"] = str(man)
                    if req.flac:
                        files["flac"] = str(write_flac(out_dir / f"{p.clip_id}.flac", out["mix"]))
                done[p.clip_id] = {"ok": bool(out["ok"]), "gate": out["gate"], "error": out["error"],
                                   "render": out.get("info"), "files": files}

            self.renderer.render_many([RenderJob(p.clip_id, p.canon, protocol.brief_dict(p.item.brief))
                                       for p in todo], on_done)
        items = []
        for p in prepared:
            d = done.get(p.clip_id) or {"ok": False, "gate": p.gate, "error": p.detail, "render": None, "files": {}}
            items.append({"i": p.item.i, "clip_id": p.clip_id, "dup_of": dup_of[p.item.i], **d})
        self._count(requests_render=1)
        return {"protocol": PROTOCOL, **self.versions, "items": items,
                "t_ms": {"total": round(1000 * (time.perf_counter() - t0), 2)}}


def request_error(err: Exception) -> tuple[int, str]:
    """The HTTP status and message of an exception raised while handling a request."""
    if isinstance(err, ProtocolError):
        return 400, str(err)
    if isinstance(err, RequestError):
        return err.status, str(err)
    return 500, f"{type(err).__name__}: {err}"
