"""Aggregates for a finished batch (docs/RATING.md; docs/design/taste_pipeline.md §1, mode `audit`).

Counts and rates over his own votes only: heard and anchor rates, repeat agreement with a Wilson interval,
position-of-best chi-square, time spent against the daily budget, and per mode the tallies (shootout options,
tiers, hearts, tags). The per-round rows for replay carry positions and his picks, never a hidden label, a
real clip id or anything from a model. The server serves this only once every round of the batch has a vote.

Stdlib only.
"""
from __future__ import annotations

import math
from datetime import date, datetime

from lofi_rl.ui.manifest import Hidden, Manifest
from lofi_rl.ui.votes import VOTE_TYPES, active_votes

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval for k successes out of n."""
    if n <= 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (series below a+1, continued fraction above)."""
    if x <= 0:
        return 1.0
    log_pre = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        ap, s = a, 1.0 / a
        term = s
        for _ in range(1000):
            ap += 1
            term *= x / ap
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(log_pre))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(log_pre) * h)


def chi2_sf(x: float, df: int) -> float:
    """P(chi-square with df degrees of freedom >= x)."""
    if df <= 0:
        raise ValueError("df must be positive")
    return _gammaincc(df / 2.0, x / 2.0)


def chi2_uniform(counts: list[int]) -> tuple[float | None, float | None]:
    n, k = sum(counts), len(counts)
    if n == 0 or k < 2:
        return None, None
    e = n / k
    x2 = sum((o - e) ** 2 / e for o in counts)
    return x2, chi2_sf(x2, k - 1)


def heard_bars(c: dict) -> int:
    return str(c.get("heard_bars", "")).count("1")


def heard_over_attempts(events: list[dict]) -> dict[tuple, str]:
    """(round, clip) -> its heard bars OR-ed over every attempt of that round in `events`: kept and undone votes
    and abandoned rounds. An undo or a reload starts the round's bar count again on the page, but what he heard
    before still counts (Shresth, 2026-10-31; D-013 as amended)."""
    out: dict[tuple, str] = {}
    for e in events:
        if e.get("type") not in VOTE_TYPES and e.get("type") != "abandon":
            continue
        for c in e.get("clips") or []:
            key, bars = (e.get("round"), c.get("clip")), str(c.get("heard_bars", ""))
            prev = out.get(key, "")
            n = max(len(prev), len(bars))
            out[key] = "".join("1" if "1" in (prev[i:i + 1], bars[i:i + 1]) else "0" for i in range(n))
    return out


def with_merged_heard(v: dict, heard: dict[tuple, str]) -> dict:
    """A copy of vote `v` whose clips carry the heard bars of every attempt of its round (heard_over_attempts)."""
    clips = [dict(c, heard_bars=heard.get((v.get("round"), c.get("clip")), c.get("heard_bars", "")))
             for c in v.get("clips") or []]
    return dict(v, clips=clips)


def implied_pairs(mode: str, v: dict, heard_min: int) -> dict[frozenset, str]:
    """Strict pairs a vote implies, over clips heard for at least `heard_min` bars: {frozenset(a, b): winner}."""
    clips = v.get("clips") or []
    ids = [c["clip"] for c in clips]
    heard = {c["clip"] for c in clips if heard_bars(c) >= heard_min}
    out: dict[frozenset, str] = {}
    if mode == "ab":
        p = v.get("pick")
        for x in ids:
            if x != p and p in heard and x in heard:
                out[frozenset((p, x))] = p
    elif mode == "keep3":
        b = v.get("best")
        if b in heard:
            for x in heard - {b}:
                out[frozenset((b, x))] = b
    elif mode in ("bws4", "shootout"):
        b, w = v.get("best"), v.get("worst")
        if b in heard:
            for x in heard - {b}:
                out[frozenset((b, x))] = b
        if w in heard:
            for x in heard - {w}:
                out.setdefault(frozenset((x, w)), x)
    return out


def agreement(mode: str, first: dict, second: dict, heard_min: int) -> tuple[int, int]:
    """(agreeing, compared) between two presentations of the same clips."""
    if mode == "triage":
        a, b = first.get("tier"), second.get("tier")
        return (int(a == b), 1) if a and b else (0, 0)
    if mode == "playlist":
        return int(first.get("mark") == second.get("mark")), 1
    pa, pb = implied_pairs(mode, first, heard_min), implied_pairs(mode, second, heard_min)
    common = pa.keys() & pb.keys()
    return sum(pa[k] == pb[k] for k in common), len(common)


def anchor_caught(mode: str, v: dict, anchor: str | None) -> bool:
    """Did he treat the broken copy as broken? (worst in best/worst rounds, Toss in triage, not picked or kept)"""
    if mode in ("bws4", "shootout"):
        return anchor is not None and v.get("worst") == anchor
    if mode == "triage":
        return v.get("tier") == "toss"
    if mode == "ab":
        return v.get("pick") not in (None, anchor)
    if mode == "keep3":
        return v.get("best") != anchor
    if mode == "playlist":
        return v.get("mark") == "cross"
    return False


def best_position(mode: str, v: dict) -> int | None:
    chosen = v.get("pick") if mode == "ab" else v.get("best")
    for c in v.get("clips") or []:
        if chosen is not None and c["clip"] == chosen:
            return c["pos"]
    return None


def _ms(e: dict) -> int:
    t = e.get("t_ms") or {}
    return int(t.get("commit") or t.get("abandon") or 0)


def _day(e: dict) -> date | None:
    try:
        return datetime.fromisoformat(e.get("ts")).date()
    except (TypeError, ValueError):
        return None


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def aggregate(m: Manifest, h: Hidden, events: list[dict], *, all_events: dict | None = None,
              heard_min_bars: int = 1, toss_min_bars: int = 1, daily_budget_min: float = 60.0,
              today: date | None = None) -> dict:
    """Everything the audit page shows for one batch. Heard bars are merged over every attempt of a round."""
    mode = m.mode
    today = today or datetime.now().astimezone().date()
    all_events = dict(all_events or {})
    all_events.setdefault(m.batch, events)
    act = active_votes(events)
    heard = heard_over_attempts(events)
    votes = [with_merged_heard(act[r.id], heard) for r in m.rounds if r.id in act]
    out: dict = {"batch": m.batch, "mode": mode, "rounds": len(m.rounds), "voted": len(votes),
                 "retracted": sum(e.get("type") == "retract" for e in events),
                 "abandoned": sum(e.get("type") == "abandon" for e in events)}

    total = ok = 0
    for v in votes:
        need = toss_min_bars if mode == "triage" and v.get("tier") == "toss" else heard_min_bars
        for c in v.get("clips") or []:
            total += 1
            ok += heard_bars(c) >= need
    out["heard"] = {"clips": total, "heard": ok, "rate": _rate(ok, total), "min_bars": heard_min_bars}

    anchors = []
    for v in votes:
        hid = v.get("hidden") or {}
        if hid.get("kind") == "anchor":
            anchors.append((v, anchor_caught(mode, v, hid.get("anchor"))))
    groups: dict[tuple, dict] = {}
    for v, caught in anchors:
        g = groups.setdefault((v.get("session"), v.get("micro")),
                              {"session": v.get("session"), "micro": v.get("micro"), "anchors": 0, "caught": 0})
        g["anchors"] += 1
        g["caught"] += int(caught)
    for g in groups.values():
        g["flag"] = g["anchors"] >= 2 and g["caught"] < g["anchors"] / 2
    caught_n = sum(int(c) for _, c in anchors)
    out["anchors"] = {"rounds": len(anchors), "caught": caught_n, "rate": _rate(caught_n, len(anchors)),
                      "micro": sorted(groups.values(), key=lambda g: (str(g["session"]), g["micro"] or 0))}

    originals: dict[str, dict] = {}
    for _batch, evs in sorted(all_events.items()):
        merged = heard_over_attempts(evs)
        originals.update({k: with_merged_heard(v, merged) for k, v in active_votes(evs).items()})
    found = missing = agree = compared = 0
    for v in votes:
        hid = v.get("hidden") or {}
        if hid.get("kind") != "repeat":
            continue
        first = originals.get(hid.get("of"))
        if first is None:
            missing += 1
            continue
        found += 1
        a, n = agreement(mode, first, v, heard_min_bars)
        agree += a
        compared += n
    lo, hi = wilson(agree, compared)
    out["repeats"] = {"rounds": found + missing, "found": found, "missing": missing, "compared": compared,
                      "agree": agree, "p": _rate(agree, compared), "ci95": [lo, hi]}

    if mode in ("bws4", "shootout", "keep3", "ab"):
        by_k: dict[int, list[int]] = {}
        for v in votes:
            pos = best_position(mode, v)
            if pos is not None:
                k = len(v.get("clips") or [])
                by_k.setdefault(k, [0] * k)[pos - 1] += 1
        rows = []
        for k, counts in sorted(by_k.items()):
            x2, p = chi2_uniform(counts)
            rows.append({"clips": k, "counts": counts, "chi2": x2, "df": k - 1, "p": p})
        out["position_of_best"] = rows

    batch_ms = sum(_ms(e) for e in events if e.get("type") in VOTE_TYPES or e.get("type") == "abandon")
    today_ms = sum(_ms(e) for evs in all_events.values() for e in evs
                   if (e.get("type") in VOTE_TYPES or e.get("type") == "abandon") and _day(e) == today)
    out["time"] = {"minutes": round(batch_ms / 60000, 1), "est_min": m.est_min,
                   "today_minutes": round(today_ms / 60000, 1), "daily_budget_minutes": daily_budget_min}

    if mode in ("bws4", "shootout"):
        out[mode] = {"no_best": sum(bool(v.get("no_best")) for v in votes),
                     "no_worst": sum(bool(v.get("no_worst")) for v in votes),
                     "all_bad": sum(bool(v.get("all_bad")) for v in votes)}
    if mode == "bws4":
        tags: dict[str, int] = {}
        for v in votes:
            name = (v.get("tag") or {}).get("name")
            if name:
                tags[name] = tags.get(name, 0) + 1
        out["bws4"]["tags"] = tags
    if mode == "shootout":
        opts: dict[str, dict] = {}
        category = None
        for v in votes:
            category = category or v.get("category")
            if (v.get("hidden") or {}).get("kind") == "repeat":
                continue          # a hidden repeat measures self-consistency; counting it again unbalances the blocks
            anchor = (v.get("hidden") or {}).get("anchor")
            for c, opt in zip(v.get("clips") or [], v.get("options") or []):
                if opt is None or c["clip"] == anchor:
                    continue
                o = opts.setdefault(opt, {"option": opt, "shown": 0, "best": 0, "worst": 0})
                o["shown"] += 1
                o["best"] += int(v.get("best") == c["clip"])
                o["worst"] += int(v.get("worst") == c["clip"])
        for o in opts.values():
            o["net"] = o["best"] - o["worst"]
        out["shootout"]["category"] = category or h.category
        out["shootout"]["options"] = sorted(opts.values(), key=lambda o: (-o["net"], -o["best"], o["option"]))
    elif mode == "keep3":
        out["keep3"] = {"kept": sum(v.get("best") is not None for v in votes),
                        "none_acceptable": sum(bool(v.get("all_bad")) for v in votes)}
    elif mode == "triage":
        tiers = {t: sum(v.get("tier") == t for v in votes) for t in ("toss", "fine", "good", "great")}
        out["triage"] = {"tiers": tiers, "fine_or_better": _rate(len(votes) - tiers["toss"], len(votes))}
    elif mode == "ab":
        out["ab"] = {"picks": [sum(best_position(mode, v) == i for v in votes) for i in (1, 2)]}
    elif mode == "playlist":
        hearts = sum(v.get("mark") == "heart" for v in votes)
        out["playlist"] = {"played": len(votes), "hearts": hearts,
                           "crosses": sum(v.get("mark") == "cross" for v in votes),
                           "heart_rate": _rate(hearts, len(votes))}

    rows = []
    for r in m.rounds:
        v = act.get(r.id)
        pos = {c["clip"]: c["pos"] for c in (v.get("clips") or [])} if v else {}
        heard = {c.get("oid"): c.get("heard_bars") for c in (v.get("clips") or [])} if v else {}
        picks: dict = {}
        if v:
            if mode in ("bws4", "shootout"):
                picks = {"best": pos.get(v.get("best")), "worst": pos.get(v.get("worst")),
                         "no_best": bool(v.get("no_best")), "no_worst": bool(v.get("no_worst")),
                         "all_bad": bool(v.get("all_bad"))}
                if mode == "bws4":
                    picks["tag"] = (v.get("tag") or {}).get("name")
            elif mode == "keep3":
                picks = {"kept": pos.get(v.get("best")), "none_acceptable": bool(v.get("all_bad"))}
            elif mode == "triage":
                picks = {"tier": v.get("tier")}
            elif mode == "ab":
                picks = {"pick": pos.get(v.get("pick"))}
            elif mode == "playlist":
                picks = {"mark": v.get("mark"), "plays": v.get("plays")}
        rows.append({"round": r.id, "number": r.number, "scene": r.scene, "voted": v is not None,
                     "event_id": v["event_id"] if v else None, "order": v.get("index") if v else None,
                     "clips": [{"pos": c.pos, "id": c.id, "file": c.file, "heard_bars": heard.get(c.id)}
                               for c in r.clips],
                     "picks": picks})
    out["list"] = rows
    return out
