"""The shootout tally (docs/SHOOTOUTS.md section 7.9, with section 8's rules). Laptop only.

    python -m lofi_rl.shootout.tally --votes data/votes --inbox inbox [--batch <name> ...] [--out <dir>]
        [--report <file.md>] [--plan shootouts/plan.yaml] [--config configs/rating.yaml] [--levels <levels.json> ...]
        [--runs runs] [--day2-cuts none|<cut>+<cut> [--day2-cuts-by <his words, date>]]
        [--serve-date YYYY-MM-DD] [--day3-cuts none|<cut>+<cut> [--day3-cuts-by <his words, date>]]

It reads each shootout batch's vote log (`<votes>/<batch>.jsonl`, the rating page's events: docs/RATING.md Part 2)
and its manifests (`<inbox>/<date>/<batch>/`), reruns `ui.audit.aggregate` on each batch, and writes
`<out>/<batch>.json` and `<out>/shootouts.json` (default `<votes>/audit/`: private, never committed). Per stage,
across its batches (section 8):

- the audit's net (best - worst over every shown clip, an anchor round's broken clip skipped, hidden repeats left
  out) and Claude's recount [CD-38]: BW = (best - worst) / heard, over appearances heard for `heard.min_bars` bars or
  more (1: D-013 as amended, Shresth 2026-10-31), a clip's heard bars OR-ed over every attempt of its round in that
  batch's log (kept and undone votes and abandoned attempts: ui.audit's heard_over_attempts), so only a mark on a clip
  that never played is left out; anchor rounds dropped; every vote of a micro-session that missed its anchor at half
  weight (D-013);
- T and B add no mark; G marks every clip of its round all-bad, and an option all-bad in half its rounds or more
  cannot survive [CD-44];
- ties [CD-39]: BW, then more bests, then fewer worsts; on a single-winner choice the default if it is among the
  tied, otherwise Shresth's word;
- repeats scored for agreement (implied pairs, the audit's figure, pooled with a Wilson 95 % interval) and left out
  of the counts [CD-47], except a repeat voted fewer calendar days after its first showing's vote than D-013's
  `controls.repeats.min_days_after` (each vote's date in its own offset), which goes to `too_soon` and is not
  scored; position-of-best counts pooled by clip count over the shootout batches (descriptive);
- rounds flagged by section 3, from the hidden manifests' levels: a round whose candidates' bus loudness (`bus_lufs`)
  spreads over 1.0 LU, over 0.5 LU for the timing variants [CD-26], and a vamp round whose keys (`bus_lufs`) or bass
  (`bass_lufs`) loudness spreads over 1.0 LU [CD-27]; from Day 2 on every other bus a stage matches too
  (`<bus>_lufs`, CD-53: R3a's sax, keys and bass, R3b's keys and bass), at its stage's limit; on Day 3 a W1 round
  whose excerpts' beds, measured in the finished renders relative to their music (`bed_lu`), spread over 1.0 LU (W2's
  level is its candidate), and a W1 or W2 anchor round whose sax as rendered (`sax_lufs`: its sour copy plays its own
  fader, matched to its round-mates') spreads over 1.0 LU; with --levels, every trim or fader a job's drift check
  moved by more than 0.1 dB.

Section 8's rules are the functions `keys_to_k2`, `kits_after_r1`, `bass_pairings`, `relisten_bass` (CD-54),
`sax_to_final` and `day2_cut` (after Day 1), `keys_survivors`, `sax_final`, `warmth`, `timing`, `swing`, `humanize`,
`vamp_ranks` and `bass_after_relisten` (after Day 2), and `rain_excerpt` and `rain_level` (after Day 3), each applied
once every round of its stage's batches has a vote (an anchor anywhere in them can halve a micro-session that began
earlier). After Day 1 they make job 2's `--leaders` string (section 7.7), checked with `draw.parse_leaders` and the
Day-2 band the draw builds (`draw.day_band`, after the adds and section 2.7's cuts), and its ARGS add `--prior`: the
job whose pulled summary (--runs, runs/shootout-<job>/summary/batches.json) built the Day-1 batches, since job 2
compares its drift keys with job 1's (section 3); `--add R2r` when the bass is undecided (CD-54: the kit slot then
carries the surviving kits, leader first, and the bass slot the default meanwhile); and `--cut`, from the minutes of
the Day-2 plan the draw will play with those leaders (`draw.day_minutes`), or, with `--day2-cuts`, from Shresth's
word on Day 2's cuts, which overrides the list: job 2 runs his cuts, the S1 rule keeps the final unless he cuts it,
and the list's cuts stay in the report as advice. After Day 2 they make job 3's line the same way (`_day3`): the
Day-2 picks as `--leaders`, every variant slot named, from the rules and job 2's pulled summary/leaders.json (what
job 2 ran: its kits, the sax S2 ran on, its adds and cuts); `--prior` job 1 and job 2; `--repeats`, the stages whose
design rounds were all voted at least two days before the serving date Claude gives with --serve-date
(`first_votes`); `--serve-date` itself, which job 3 records and writes into every Day-3 hidden manifest as
`serve_from`, so `make rate` serves no Day-3 batch before it; `--rerun-s2` when the final picks the other finalist;
and Day 3's cut (`day3_cut`, or his word with --day3-cuts). A case the spec does not decide raises
`RuleGap`, which names the section; the tally records it and lists it with everything else that needs Shresth's
word. A batch with no votes is reported as such, never as zeros. A seeded draw (CD-41, CD-18, a provisional leader) is keyed by the stage's
standing vote ids, so it is fixed once the votes exist and the private logs reproduce it.

--report writes the tables, the flags, the repeats and the proposal as markdown, which Claude pastes into
gates/GATE_palette.md after Shresth confirms (section 1 step 2). The report holds counts only: per option, never per
tile, so it gives away no draw. Nothing here says how anything sounds.

Inside the repo, --out and --report are written only under data/ (gitignored), as the votes are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from lofi_rl.ui.audit import (aggregate, agreement, anchor_caught, chi2_uniform, heard_bars, heard_over_attempts,
                              wilson, with_merged_heard)
from lofi_rl.ui.manifest import ManifestError, load_batch, load_public
from lofi_rl.ui.serve import ROOT as REPO
from lofi_rl.ui.serve import ConfigError, _inside, load_config
from lofi_rl.ui.votes import active_votes, atomic_write, read_all, read_events

from .draw import DrawError, Rng, day_band, day_minutes, parse_leaders, round_seconds
from .plan import BAND_SLOTS, BUS_SLOT, MIX_ORDER, PLAN, Plan, PlanError, apply_adds, apply_cuts, load_plan

SCHEMA = "lofi-rl/shootout-tally/1"
BATCH_SCHEMA = "lofi-rl/shootout-tally-batch/1"
ONE, HALF, ZERO = Fraction(1), Fraction(1, 2), Fraction(0)
MATCHED_LU = 1.0              # section 3 step 8: an instrument round's candidates; CD-27: a vamp round's keys and bass
TIMING_LU = 0.5               # section 3 "Not matched": the variants that move notes in time [CD-26]
TIMING_SLOTS = frozenset({"swing", "humanize", "timing"})
UNMATCHED_SLOTS = frozenset({"bed", "rain"})            # rain: the level is the candidate
BED_SLOT = "bed"              # W1: the excerpt is the candidate at one level, so its beds are compared like a bus
MOVE_DB = 0.1                 # section 3 "Drift between jobs": a trim or fader that moves more is listed
DAY2_PUSHES_MIN = 30.0        # section 2.7 cut list: Day 2's two pushes of 15 minutes
DAYS12_MIN = 55.0             # section 2.7 cut list: P section 7's "about 55 min" for Nov 3-4
DAY3_MIN = 30.0               # section 2.7: "Day 3 is cut only if r x its plan would pass 30"
LEADERS_SCHEMA = "lofi-rl/shootout-leaders/1"            # a job's summary/leaders.json (lofi_rl.shootout.job)
DAY1_STAGES = ("K1", "R1", "R2", "S1")
RELISTEN = "R2r"              # Day 2's bass re-listen (CD-54): the optional stage the Day-1 proposal adds with --add
WORD = "Shresth's word"       # the record of --day2-cuts when --day2-cuts-by gives no words
RERUN = " (re-run)"           # a stage of `rerun_s2` heard again on a later day (section 4.4)
WARMTH = ("sax_plain", "sax_warm_on")                   # section 8 Sax variants: warmth is one of these two
CONTROL = "sax_expr_off"                                # its control
BORROWED_TEST = {"wistful": ("float", "last"), "melancholy": ("plagal", "first")}   # CD-45 (section 8 Vamps)
RAIN_NONE = "rain_none"
POOLS = (("K1 + K2", "K1", "K2"), ("S1 + S1f", "S1", "S1f"))
REPEAT_CAVEAT = ("What could give a repeat away (section 2.5): on Nov 6 the keys, drums + bass, vamp and sax rounds "
                 "are otherwise over, so a round that changes keys, drums or bass, chords or the sax is a repeat; a "
                 "vamp repeat is the only round that day with no sax; a repeat has no rain in any clip, unlike the "
                 "rain rounds; a Day-1 repeat plays the dev band in its fixed slots. One repeat per shootout gives 3-5 "
                 "pairs, too few to flag a single shootout (CD-47).")


class TallyError(ValueError):
    """The inputs cannot be tallied: a missing folder, an unknown batch, a recount that disagrees with the audit."""


class RuleGap(TallyError):
    """Section 8 does not decide this case. The message names the section; Shresth's word decides it."""


# ------------------------------------------------------------------------------------------------ the rows
@dataclass
class Vote:
    """One standing vote, as the recount reads it."""

    event_id: str
    ts: str | None
    session: str | None
    micro: int | None
    weight: Fraction                  # 1, or 1/2 in a micro-session that missed its anchor (D-013)
    clips: tuple[dict, ...]           # tile order: oid, real, pos, option, heard_bars (merged), own_bars, heard, kit
    best: str | None                  # real clip ids
    worst: str | None
    no_best: bool
    no_worst: bool
    all_bad: bool
    anchor: str | None                # an anchor round's broken copy (real id)
    raw: dict = field(repr=False)

    def clip(self, real: str | None) -> dict | None:
        return next((c for c in self.clips if real is not None and c["real"] == real), None)


@dataclass
class Round:
    """One round of a batch, from its hidden manifest, with its standing vote if it has one."""

    batch: str
    date: str
    id: str
    number: int                       # 1-based position in the manifest
    kind: str                         # none | anchor | repeat
    stage: str                        # the tally's stage: the clips' `stage` ("S2a (re-run)" for a later-day re-run)
    slot: str | None
    loop: str | None
    of: str | None
    infos: tuple[dict, ...] = field(repr=False)   # the hidden manifest's clip info in tile order, with oid and pos
    vote: Vote | None = None

    @property
    def kit(self) -> str | None:
        kits = {i.get("kit") for i in self.infos}
        return kits.pop() if len(kits) == 1 else None


@dataclass
class Row:
    """One option's figures in a stage (or a pool of stages). Weighted figures are Fractions (D-013 halves)."""

    option: str
    shown: int = 0                    # appearances in design rounds
    heard_n: int = 0                  # of them heard for heard.min_bars bars or more, over every attempt (D-013)
    heard: Fraction = ZERO            # the same, weighted: BW's denominator
    best: Fraction = ZERO             # marks on heard clips, weighted
    worst: Fraction = ZERO
    rounds: Fraction = ZERO           # design rounds it was shown in, weighted (CD-44's denominator)
    all_bad: Fraction = ZERO          # of them marked G, weighted
    all_bad_n: int = 0
    unheard_marks: int = 0            # best or worst marks left out: heard less in every attempt (CD-38)
    a_shown: int = 0                  # the audit's counts: anchor rounds in (their broken clip skipped), heard or not
    a_best: int = 0
    a_worst: int = 0

    @property
    def bw(self) -> Fraction | None:
        return (self.best - self.worst) / self.heard if self.heard else None

    @property
    def net(self) -> int:
        return self.a_best - self.a_worst

    @property
    def out(self) -> bool:
        """CD-44: all-bad in half its rounds or more, so it cannot survive."""
        return self.rounds > 0 and 2 * self.all_bad >= self.rounds


def tally_rows(rounds: Iterable[Round]) -> dict[str, Row]:
    """Every option's figures over the voted rounds given (section 8): the recount over design rounds and the audit's
    counts over design and anchor rounds; repeats count only for agreement."""
    rows: dict[str, Row] = {}
    for rd in rounds:
        v = rd.vote
        if v is None or rd.kind == "repeat":
            continue
        for c in v.clips:
            if c["option"] is None:
                continue
            row = rows.setdefault(c["option"], Row(c["option"]))
            if c["real"] != v.anchor:                    # ui/audit.py: an anchor round's broken clip is skipped
                row.a_shown += 1
                row.a_best += int(v.best == c["real"])
                row.a_worst += int(v.worst == c["real"])
            if rd.kind != "none":                        # the recount drops anchor rounds (CD-38)
                continue
            row.shown += 1
            row.rounds += v.weight
            if v.all_bad:                                # G marks every clip of its round (CD-44)
                row.all_bad += v.weight
                row.all_bad_n += 1
            if c["heard"]:
                row.heard_n += 1
                row.heard += v.weight
                row.best += v.weight if v.best == c["real"] else ZERO
                row.worst += v.weight if v.worst == c["real"] else ZERO
            elif c["real"] in (v.best, v.worst):
                row.unheard_marks += 1
    return rows


# ------------------------------------------------------------------------------------------------ ranking
def _key(o: Row) -> tuple:
    """CD-39's order: BW, then more bests, then fewer worsts. An option never heard has no BW and ranks last."""
    return (o.bw is None, -(o.bw if o.bw is not None else ZERO), -o.best, o.worst)


def rank(rows: Iterable[Row]) -> list[list[Row]]:
    """The options in CD-39's order, as groups of the fully tied (each group by option id)."""
    groups: list[list[Row]] = []
    for o in sorted(rows, key=lambda r: (_key(r), r.option)):
        if groups and _key(groups[-1][0]) == _key(o):
            groups[-1].append(o)
        else:
            groups.append([o])
    return groups


def _why(a: Row, b: Row) -> str:
    """What puts `a` ahead of `b` under CD-39."""
    if a.bw != b.bw:
        return "BW"
    if a.best != b.best:
        return "more bests (BW tied)"
    return "fewer worsts (BW and bests tied)"


def rank_doc(groups: Sequence[Sequence[Row]]) -> list[dict]:
    """Ranks as places: tied options share the place of the first of them."""
    out, place = [], 1
    for g in groups:
        for o in g:
            out.append({"place": place, "tied": len(g) > 1, "option": o.option, "bw": _f(o.bw),
                        "best": _f(o.best), "worst": _f(o.worst), "cannot_survive": o.out})
        place += len(g)
    return out


def vote_seed(rounds: Iterable[Round]) -> int:
    """A seed fixed by the standing votes of the design rounds: their event ids, sorted and hashed. Nobody can pick
    it after the votes, and a later repeat or anchor of the stage leaves it alone."""
    ids = sorted(rd.vote.event_id for rd in rounds if rd.vote is not None and rd.kind == "none")
    return int.from_bytes(hashlib.sha256("\n".join(ids).encode()).digest()[:8], "big")


def seeded(seed: int, label: str, pool: Sequence[str], k: int) -> dict:
    """k of `pool` by the project's SplitMix64 stream keyed by the seed and the label (draw.Rng)."""
    order = Rng.keyed(seed, "tally", label).shuffle(sorted(pool))
    return {"seed": f"{seed:016x}", "label": label, "among": sorted(pool), "picked": order[:k]}


def single_winner(rows: Iterable[Row], default: str | None, *, seed: int, label: str,
                  meanwhile: str = "default") -> dict:
    """CD-39 on a single-winner choice. The CD-39 leader among the options that can survive (CD-44); on a full tie,
    the default if it is among the tied, otherwise Shresth's word, and meanwhile the default stays (`meanwhile=
    "default"`, a palette value) or, for a slot whose leader must be one of the options (`"draw"`), a seeded draw
    names a provisional one."""
    rows = list(rows)
    eligible = [o for o in rows if not o.out]
    if not eligible:
        raise RuleGap(f"SHOOTOUTS.md section 8 (CD-39 with CD-44): no option of the {label} survives the all-bad rule "
                      f"({sorted(o.option for o in rows)}); the spec does not say what then")
    groups = rank(eligible)
    top = [o.option for o in groups[0]]
    doc = {"pick": top[0], "tied": top if len(top) > 1 else [], "default": default, "needs_word": False,
           "draw": None, "out": sorted(o.option for o in rows if o.out),
           "by": _why(groups[0][0], groups[1][0]) if len(groups) > 1 else "the only option left"}
    if len(top) == 1:
        return doc
    if default in top:
        doc.update(pick=default, by="the default among the fully tied (CD-39)")
    elif meanwhile == "default":
        doc.update(pick=default, needs_word=True,
                   by="a full tie without the default: Shresth's word, and the default stays meanwhile (CD-39)")
    else:
        d = seeded(seed, label, top, 1)
        doc.update(pick=d["picked"][0], needs_word=True, draw=d,
                   by="a full tie without the default: Shresth's word; a seeded draw names a provisional leader")
    return doc


# ------------------------------------------------------------------------------------------------ after Day 1
def keys_to_k2(rows: Mapping[str, Row], seed: int, *, default: str | None) -> dict:
    """Section 8 Keys, after K1 [CD-41]: K1's BW ranks all 8 and the top 3 go to K2; a full tie at third place takes
    every tied option up to 4, and beyond 4 a seeded draw among the fully tied fills the 4. An option all-bad in half
    its rounds cannot survive (CD-44), so it is not sent. The leader (the band's keys on Day 2, listed first) is the
    single-winner choice among the K2 participants."""
    eligible = [o for o in rows.values() if not o.out]
    out = sorted(o.option for o in rows.values() if o.out)
    if len(eligible) < 3:
        raise RuleGap(f"SHOOTOUTS.md section 8 Keys (CD-41 with CD-44): only {len(eligible)} keys survive the all-bad "
                      f"rule ({out} cannot), and K2 plays 3 or 4; the spec does not say what fills K2")
    groups = rank(eligible)
    chosen: list[str] = []
    tie = draw = None
    for g in groups:
        if len(chosen) >= 3:
            break
        names = [o.option for o in g]
        if len(chosen) + len(g) <= 3:
            chosen += names
            continue
        tie = names                                       # a full tie across third place
        if len(chosen) + len(g) <= 4:
            chosen += names
        else:
            draw = seeded(seed, "K2", names, 4 - len(chosen))
            chosen += draw["picked"]
        break
    leader = single_winner([rows[o] for o in chosen], default, seed=seed, label="K1 leader", meanwhile="draw")
    return {"rule": "section 8 Keys (CD-41)", "rank": rank_doc(rank(rows.values())), "out": out,
            "k2": [leader["pick"]] + [o for o in chosen if o != leader["pick"]], "tie_at_third": tie, "draw": draw,
            "leader": leader}


def kits_after_r1(rows: Mapping[str, Row], seed: int, *, default: str | None) -> dict:
    """Section 8 Kits [CD-42]: all 3 survive unless the last by R1 BW has BW below 0, which leaves 2; a full tie for
    last keeps all 3. A kit all-bad in half its rounds cannot survive (CD-44); with one gone, the other two stay
    (TASTE keeps 2-3, and dropping two would leave one). The leader (the Day-2 band's kit) is the single-winner
    choice among the survivors."""
    eligible = [o for o in rows.values() if not o.out]
    out = sorted(o.option for o in rows.values() if o.out)
    if len(eligible) < 2:
        raise RuleGap(f"SHOOTOUTS.md section 8 Kits (CD-42 with CD-44): {len(eligible)} kit(s) survive the all-bad "
                      f"rule ({out} cannot); TASTE keeps 2-3 kits and the spec does not say what then")
    groups = rank(eligible)
    dropped = None
    if len(eligible) >= 3 and len(groups[-1]) == 1:
        last = groups[-1][0]
        if last.bw is not None and last.bw < 0:
            dropped = last.option
    survivors = [o.option for g in groups for o in g if o.option != dropped]
    leader = single_winner([rows[k] for k in survivors], default, seed=seed, label="R1 leader", meanwhile="draw")
    return {"rule": "section 8 Kits (CD-42)", "rank": rank_doc(rank(rows.values())), "out": out, "dropped": dropped,
            "tie_for_last": len(groups[-1]) > 1 and len(eligible) >= 3, "survivors": survivors, "leader": leader}


def bass_pairings(r2: Sequence[Round], rows: Mapping[str, Row], kits: Sequence[str], seed: int, *,
                  default: str | None) -> dict:
    """Section 8 Basses [CD-43]: each surviving kit takes the best bass of its own R2 round; a round with no counted
    best (T, G, or a best on a clip that never played in any attempt, CD-38) or whose best cannot survive (CD-44)
    takes the pooled R2 leader, the single-winner choice over every R2 round. Each pairing rests on one round. When
    that leaves the bass undecided, Day 2 adds R2r (`relisten_bass`, CD-54)."""
    pooled = single_winner(rows.values(), default, seed=seed, label="R2 leader", meanwhile="draw")
    pairs = {}
    for kit in kits:
        own = [rd for rd in r2 if rd.kit == kit and rd.kind == "none"]
        if len(own) != 1:
            raise RuleGap(f"SHOOTOUTS.md section 8 Basses (CD-43): kit {kit} has {len(own)} R2 rounds, and the rule "
                          "takes the best of its own one round")
        rd, v = own[0], own[0].vote
        best = v.clip(v.best) if v is not None else None
        if best is not None and best["heard"] and not rows[best["option"]].out:
            pairs[kit] = {"bass": best["option"], "from": "its own R2 round", "slot": rd.slot, "loop": rd.loop}
            continue
        why = ("no vote" if v is None else "no clear best (T)" if v.no_best else "all bad (G)" if v.all_bad
               else "its best never played in any attempt" if best is not None and not best["heard"]
               else "its best cannot survive (CD-44)")
        pairs[kit] = {"bass": pooled["pick"], "from": "the pooled R2 leader", "why": why, "slot": rd.slot,
                      "loop": rd.loop}
    return {"rule": "section 8 Basses (CD-43)", "rank": rank_doc(rank(rows.values())), "pooled_leader": pooled,
            "pairings": pairs}


def sax_to_final(rows: Mapping[str, Row], s1: Sequence[Round], seed: int, *, default: str | None,
                 final: bool = True) -> dict:
    """Section 8 Sax, after S1 [CD-18]: the S1 BW top 2 go to the final. A full tie for the places (after CD-39's
    order) is settled by a seeded draw; three or more fully tied give a three-source final, which a seeded draw fills
    when more are tied than places are left. The leader (S2 runs on it, listed first) is the single-winner choice
    among the finalists. With the `final` cut, S1 alone picks the sax: the single-winner choice over S1. If the S1
    leader is all-bad in half its rounds, or most S1 rounds are all-bad, no free sax passes and D-012's fallback
    (the lead becomes Rhodes) is Shresth's call."""
    design = [rd for rd in s1 if rd.kind == "none" and rd.vote is not None]
    w_all = sum((rd.vote.weight for rd in design), ZERO)
    w_bad = sum((rd.vote.weight for rd in design if rd.vote.all_bad), ZERO)
    raw_top = rank(rows.values())[0]
    no_free = []
    if any(o.out for o in raw_top):
        no_free.append(f"the S1 leader by BW ({', '.join(o.option for o in raw_top)}) is all-bad in half its rounds "
                       "or more")
    if w_all and 2 * w_bad > w_all:
        no_free.append(f"most S1 rounds are all-bad ({_n(w_bad)} of {_n(w_all)})")
    doc: dict[str, Any] = {"rule": "section 8 Sax (CD-18)", "rank": rank_doc(rank(rows.values())),
                           "out": sorted(o.option for o in rows.values() if o.out), "no_free_sax": no_free,
                           "final": final, "tie": None, "draw": None, "three_source": False}
    eligible = [o for o in rows.values() if not o.out]
    if not final:
        doc["leader"] = single_winner(rows.values(), default, seed=seed, label="S1 leader", meanwhile="draw")
        doc["sax"] = [doc["leader"]["pick"]]
        return doc
    if len(eligible) < 2:
        raise RuleGap(f"SHOOTOUTS.md section 8 Sax (CD-18 with CD-44): {len(eligible)} source(s) survive the all-bad "
                      "rule and the final takes 2; the spec does not say what then (D-012's fallback is Shresth's "
                      "call)")
    chosen: list[str] = []
    for g in rank(eligible):
        if len(chosen) >= 2:
            break
        names = [o.option for o in g]
        if len(chosen) + len(g) <= 2:
            chosen += names
            continue
        doc["tie"] = names
        need = 2 - len(chosen)
        if len(g) >= 3:                                   # three or more tied: a three-source final
            doc["three_source"] = True
            need = 3 - len(chosen)
        if len(g) == need:
            chosen += names
        else:
            doc["draw"] = seeded(seed, "S1 final", names, need)
            chosen += doc["draw"]["picked"]
        break
    doc["leader"] = single_winner([rows[o] for o in chosen], default, seed=seed, label="S1 leader",
                                  meanwhile="draw")
    doc["sax"] = [doc["leader"]["pick"]] + [o for o in chosen if o != doc["leader"]["pick"]]
    return doc


def day2_cut(plan: Plan, day1_minutes: float, *, counts: Mapping[str, int] | None = None,
             adds: Sequence[str] = (), decided: Sequence[str] | None = None, by: str | None = None) -> dict:
    """Section 2.7's cut list [CD-50]: r = the audit's Day-1 minutes / the plan's (23.0, the Day-1 batches'
    est_min). Day 2's plan is the estimate of the rounds the Day-2 draw will play with the proposal's leaders
    (`counts`: K2's 3 or 4 keys, the final's 2 or 3 saxes) and adds (R2r, CD-54), from `draw.day_minutes`: every
    design round exactly, each anchor at the mean of the rounds it may imitate (28.2 min with the default counts and
    no R2r; section 6's fixed batch figures sum to 28.3, with a longer sax-batch anchor). If r x Day 2's plan would
    pass 30 or r x Days 1-2's would pass 55, cut in the plan's order until both fit, each cut's saving the drop in
    that estimate (the drawn rounds it removes, and its anchor's change). R2r is never cut: it is not on the list.
    `decided` is Shresth's word on Day 2's cuts (`decided_cuts`, the CLI's --day2-cuts), which overrides the list:
    the result's `cuts` stay the list's, as advice, and `decided` gives his cuts, `by` (his words), and the same
    minutes at his cuts."""
    p1 = round(sum(b.est_min_plan for b in plan.batches if b.day == 1), 1)
    r = day1_minutes / p1 if p1 else 0.0
    counts = dict(counts or {})

    def p2(cuts: Sequence[str]) -> float:
        return day_minutes(plan, 2, counts, cuts, adds)["seconds"] / 60.0

    def fits(m: float) -> bool:
        return r * m <= DAY2_PUSHES_MIN and r * (p1 + m) <= DAYS12_MIN

    base = p2([])
    cuts, m = [], base
    for c in sorted((c for c in plan.cuts if c.day == 2), key=lambda c: c.order):
        if fits(m):
            break
        cuts.append(c.id)
        m = p2(cuts)
    doc = {"rule": "section 2.7 cut list (CD-50)", "day1_minutes": round(day1_minutes, 1), "day1_plan": p1,
           "day2_plan": round(base, 1), "counts": counts, "adds": list(adds), "r": round(r, 4), "cuts": cuts,
           "saves_min": round(base - m, 1), "day2_minutes": round(r * m, 1), "days12_minutes": round(r * (p1 + m), 1),
           "fits": fits(m), "decided": None}
    if decided is not None:                 # his word: job 2 runs these, whatever the list says
        md = p2(list(decided))
        doc["decided"] = {"cuts": list(decided), "by": by or WORD, "saves_min": round(base - md, 1),
                          "day2_minutes": round(r * md, 1), "days12_minutes": round(r * (p1 + md), 1),
                          "fits": fits(md)}
    return doc


def decided_cuts(plan: Plan, cuts: Sequence[str], day: int = 2) -> list[str]:
    """Shresth's word on a day's cuts (the CLI's --day2-cuts or --day3-cuts; [] for none): cuts of that day of the
    plan, each named once, returned in the list's order (section 2.7). TallyError otherwise."""
    ids = [c.id for c in sorted((c for c in plan.cuts if c.day == day), key=lambda c: c.order)]
    bad = [c for c in cuts if c not in ids]
    if bad:
        raise TallyError(f"--day{day}-cuts {bad}: not Day-{day} cuts of the plan ({'+'.join(ids)}; 'none' for no "
                         "cut)")
    if len(set(cuts)) != len(cuts):
        raise TallyError(f"--day{day}-cuts names a cut twice: {'+'.join(cuts)}")
    return [c for c in ids if c in cuts]


def _counted_best(rd: Round) -> dict | None:
    """The clip a round's standing vote marked best, when that clip played (over every attempt, D-013); else None."""
    v = rd.vote
    c = v.clip(v.best) if v is not None else None
    return c if c is not None and c["heard"] else None


def relisten_bass(r2: Sequence[Round], bass: Mapping, kits: Mapping, plan: Plan, *, default: str | None) -> dict:
    """CD-54 (Shresth, 2026-10-31, "Yes, use these" to the 3-round bass re-listen): after Day 1, Day 2 adds R2r when
    no R2 round has a counted best, or when a surviving kit takes the pooled R2 leader (CD-43) and the pooled leaders
    are fully tied without the default (CD-39): the bass is then undecided. The Day-2 band keeps the default bass
    meanwhile ("the default stays meanwhile", CD-39), and R2r plays the surviving kits by rank, the R1 leader first,
    as many as its slots name (2: the leader in two rounds, the other in one); a third survivor plays no R2r round.
    `bass` is bass_pairings' result, `kits` kits_after_r1's. A pooled tie no kit takes decides nothing, so it adds
    nothing."""
    st = plan.stages.get(RELISTEN)
    design = [rd for rd in r2 if rd.kind == "none"]
    with_best = [rd.slot for rd in design if _counted_best(rd) is not None]
    why = []
    if not with_best:
        why.append("no R2 round has a best")
    pooled = bass["pooled_leader"]
    takers = sorted(k for k, x in bass["pairings"].items() if x["from"] == "the pooled R2 leader")
    if takers and pooled.get("needs_word"):
        why.append(f"the pooled R2 leaders {pooled['tied']} tie without the default ({default}), and {takers} take the "
                   "pooled leader")
    doc: dict[str, Any] = {"rule": "CD-54 (section 8 Basses)", "stage": RELISTEN, "adds": bool(why) and st is not None,
                           "why": why, "rounds_with_a_best": with_best, "takers": takers, "kits": [], "left_out": [],
                           "bass": None}
    if not doc["adds"]:
        return doc
    lead = kits["leader"]["pick"]
    order = [lead] + [k for k in kits["survivors"] if k != lead]
    doc.update(kits=order[:st.kit_ranks], left_out=order[st.kit_ranks:], bass=default,
               rounds={k: [x.id for x in st.slots if order[x.kit_rank - 1] == k] for k in order[:st.kit_ranks]})
    return doc


def _judged(row: Row, by: Row) -> Row:
    """`row` whose all-bad figures (CD-44) are those of `by`: a kit's own R2r rounds are judged over every R2r round."""
    return replace(row, rounds=by.rounds, all_bad=by.all_bad, all_bad_n=by.all_bad_n)


def bass_after_relisten(r2r: Sequence[Round], kits: Sequence[str], *, default: str | None) -> dict:
    """Section 8 Basses after R2r (CD-43 as CD-54 reads it): each surviving kit takes the best bass of its own R2r
    rounds, the CD-39 leader over them among the basses that can survive over every R2r round (CD-44); a kit whose R2r
    rounds have no counted best (T, G, a best on a clip that never played, or one that cannot survive), or that plays
    none, takes the pooled R2r leader, the CD-39 leader over every R2r round. Ties go to the default if it is among
    the tied, else to Shresth's word, the default staying meanwhile. R2's votes are not pooled in: R2r decides."""
    design = [rd for rd in r2r if rd.kind == "none"]
    pooled_rows = tally_rows(design)
    pooled = single_winner(pooled_rows.values(), default, seed=0, label="R2r leader", meanwhile="default")
    pairs = {}
    for kit in kits:
        own = [rd for rd in design if rd.kit == kit]
        entry = {"slots": [rd.slot for rd in own], "loops": [rd.loop for rd in own]}
        counted = [rd.slot for rd in own
                   if (c := _counted_best(rd)) is not None and not pooled_rows[c["option"]].out]
        if not counted:
            why = ("it plays no R2r round" if not own else
                   "its R2r rounds have no counted best (T, G, a best on a clip that never played, or one that cannot "
                   "survive)")
            pairs[kit] = {**entry, "bass": pooled["pick"], "from": "the pooled R2r leader", "why": why, "pick": pooled}
            continue
        rows = {o: _judged(x, pooled_rows[o]) for o, x in tally_rows(own).items() if not pooled_rows[o].out}
        pick = single_winner(rows.values(), default, seed=0, label=f"R2r on {kit}", meanwhile="default")
        pairs[kit] = {**entry, "bass": pick["pick"], "from": "its own R2r rounds", "best_in": counted, "pick": pick,
                      "rank": rank_doc(rank(rows.values()))}
    return {"rule": "section 8 Basses after R2r (CD-43, CD-54)", "rank": rank_doc(rank(pooled_rows.values())),
            "pooled_leader": pooled, "pairings": pairs}


def leaders_string(leaders: Mapping[str, Sequence[str]]) -> str:
    """The job's --leaders form (section 7.7): `<slot>=<id>[+<id>...]` per band slot, joined by `/`, leader first."""
    return "/".join(f"{s}={'+'.join(leaders[s])}" for s in BAND_SLOTS if s in leaders)


# ------------------------------------------------------------------------------------------------ after Day 2
def keys_survivors(rows: Mapping[str, Row], participants: Sequence[str], seed: int, *, default: str | None) -> dict:
    """Section 8 Keys, after K2 [CD-40]: survivors come only from the K2 participants, by BW over K1 + K2 (`rows`
    pooled over both stages): the top 2, and the third if its BW is above 0. CD-44 holds over K1 + K2. The leader
    (Day 3's keys) is the single-winner choice among the survivors. The spec gives no rule for a full tie across
    the survivor line, or for fewer than 2 survivors: RuleGap."""
    part = [rows[p] for p in participants if p in rows]
    eligible = [o for o in part if not o.out]
    out = sorted(o.option for o in part if o.out)
    if len(eligible) < 2:
        raise RuleGap(f"SHOOTOUTS.md section 8 Keys (CD-40 with CD-44): {len(eligible)} K2 participant(s) survive the "
                      f"all-bad rule ({out} cannot); TASTE keeps 2-3 keys and the spec does not say what then")
    survivors: list[str] = []
    for g in rank(eligible):
        names = [o.option for o in g]
        above_zero = g[0].bw is not None and g[0].bw > 0
        if len(survivors) + len(g) <= 2:
            survivors += names
            continue
        if len(survivors) < 2:                           # a full tie across the second place
            if above_zero and len(survivors) + len(g) <= 3:
                survivors += names
                break
            raise RuleGap(f"SHOOTOUTS.md section 8 Keys (CD-40): {names} are fully tied (BW, bests, worsts) across the "
                          "line of the top 2 survivors, and the spec gives no rule for a tie there")
        if above_zero:                                   # the third place, kept if its BW is above 0
            if len(g) > 1:
                raise RuleGap(f"SHOOTOUTS.md section 8 Keys (CD-40): {names} tie for the third survivor place with BW "
                              "above 0, and the spec gives no rule for a tie there")
            survivors += names
        break
    leader = single_winner([rows[s] for s in survivors], default, seed=seed, label="keys leader over K1 + K2",
                           meanwhile="draw")
    return {"rule": "section 8 Keys (CD-40)", "participants": list(participants), "rank": rank_doc(rank(part)),
            "out": out, "survivors": [leader["pick"]] + [s for s in survivors if s != leader["pick"]],
            "leader": leader}


def sax_final(final: Mapping[str, Row], pooled: Mapping[str, Row], finalists: Sequence[str], *,
              s1_leader: str | None) -> dict:
    """Section 8 Sax, after the final [CD-18]: the winner has more bests over the final's rounds (on heard clips,
    weighted); a tie goes to the higher combined S1 + final BW (`pooled`), then to Shresth. A finalist all-bad in
    half its rounds over S1 + the final cannot survive (CD-44). S2 re-runs on Day 3 with the winner if it is not
    the S1 leader S2 ran on."""
    eligible = [f for f in finalists if not pooled[f].out]
    doc: dict[str, Any] = {"rule": "section 8 Sax (CD-18)", "finalists": list(finalists),
                           "out": [f for f in finalists if pooled[f].out],
                           "final_bests": {f: _f(final[f].best) for f in finalists},
                           "combined_bw": {f: _f(pooled[f].bw) for f in finalists}, "s1_leader": s1_leader,
                           "winner": None, "tied": [], "needs_word": False, "rerun_s2": None}
    if not eligible:
        raise RuleGap("SHOOTOUTS.md section 8 Sax (CD-18 with CD-44): every finalist is all-bad in half its rounds; "
                      "the spec does not say what then (D-012's fallback is Shresth's call)")
    most = max(final[f].best for f in eligible)
    top = [f for f in eligible if final[f].best == most]
    doc["by"] = "more bests in the final"
    if len(top) > 1:
        bws = {f: pooled[f].bw for f in top}
        hi = max((b for b in bws.values() if b is not None), default=None)
        top = [f for f in top if bws[f] == hi] if hi is not None else top
        doc["by"] = "the higher combined S1 + final BW (the final tied on bests)"
        if len(top) > 1:
            doc.update(tied=top, needs_word=True, by="tied on the final's bests and on combined BW: Shresth's word")
            return doc
    doc["winner"] = top[0]
    if s1_leader is not None and top[0] != s1_leader:
        doc["rerun_s2"] = top[0]
    return doc


def warmth(rows: Mapping[str, Row], seed: int, *, default: str | None) -> dict:
    """Section 8 Sax variants, warmth: the S2a BW leader of `sax_plain` and `sax_warm_on` (ties: the default,
    sax_plain). If `sax_expr_off` has the top BW in S2a (a share of it included), Claude reports it and asks
    whether expression stays on."""
    pick = single_winner([rows[o] for o in WARMTH if o in rows], default, seed=seed, label="S2a warmth")
    ctrl = rows.get(CONTROL)
    top = max((o.bw for o in rows.values() if o.bw is not None), default=None)
    ask = ctrl is not None and ctrl.bw is not None and ctrl.bw == top
    return {"rule": "section 8 Sax variants (warmth)", "rank": rank_doc(rank(rows.values())), "pick": pick,
            "control_top": ask}


def timing(rows: Mapping[str, Row], seed: int, *, default: str | None) -> dict:
    """Section 8 Sax variants, timing: the S2b BW leader; ties as CD-39, the default being grid."""
    return {"rule": "section 8 Sax variants (timing)", "rank": rank_doc(rank(rows.values())),
            "pick": single_winner(rows.values(), default, seed=seed, label="S2b timing")}


def swing(rows: Mapping[str, Row], *, default: str | None) -> dict:
    """Section 8 Swing [CD-13], advisory: swing stays at the default (0.55) unless Shresth names another option after
    seeing the R3a tally, reported with the note that two rounds cannot separate four options reliably."""
    groups = rank(rows.values())
    return {"rule": "section 8 Swing (CD-13, advisory)", "rank": rank_doc(groups), "pick": default,
            "bw_leader": [o.option for o in groups[0]] if groups else [],
            "note": "two rounds cannot separate four options reliably"}


def humanize(rows: Mapping[str, Row], seed: int, *, default: str | None) -> dict:
    """Section 8 Humanize: the R3b BW leader; ties as CD-39, the default being off."""
    return {"rule": "section 8 Humanize", "rank": rank_doc(rank(rows.values())),
            "pick": single_winner(rows.values(), default, seed=seed, label="R3b humanize")}


def vamp_ranks(rows: Mapping[str, Row], moods: Mapping[str, str]) -> dict:
    """Section 8 Vamps: per mood, the vamps ranked by BW (seed data). The borrowed-chord report [CD-45]: if `float`
    ranks 4th in wistful and `plagal` 1st in melancholy, Claude reports it and asks Shresth; nothing leaves
    chords.numerals without his words. A tie on either condition counts as not met, read here as a BW equal to
    another vamp's of its mood (TASTE's default keeps the chords)."""
    by_mood: dict[str, list[Row]] = {}
    for o, row in rows.items():
        by_mood.setdefault(moods.get(o, "?"), []).append(row)
    cond = {}
    for mood, (name, where) in BORROWED_TEST.items():
        mine = [r for r in by_mood.get(mood, []) if r.option == name]
        others = [r for r in by_mood.get(mood, []) if r.option != name]
        if not mine or not others or mine[0].bw is None or any(o.bw is None for o in others):
            cond[mood] = {"vamp": name, "needs": where, "met": False}
            continue
        me = mine[0].bw
        met = all(me < o.bw for o in others) if where == "last" else all(me > o.bw for o in others)
        cond[mood] = {"vamp": name, "needs": where, "met": met, "bw": _f(me)}
    return {"rule": "section 8 Vamps (CD-45)",
            "moods": {m: rank_doc(rank(rs)) for m, rs in sorted(by_mood.items())},
            "borrowed": {"conditions": cond, "met": all(c["met"] for c in cond.values()) and len(cond) == 2}}


def rain_excerpt(rows: Mapping[str, Row], seed: int, *, default: str | None) -> dict:
    """Section 8 Rain, excerpt: the W1 BW leader; ties go to e1 (CD-39, the default)."""
    return {"rule": "section 8 Rain (excerpt)", "rank": rank_doc(rank(rows.values())),
            "pick": single_winner(rows.values(), default, seed=seed, label="W1 excerpt")}


def rain_level(rows: Mapping[str, Row], seed: int, *, default: str | None) -> dict:
    """Section 8 Rain, level: the W2 BW leader; ties go to Δ 24 (CD-39, the default). A rain_none lead goes to
    Shresth, since no bed changes T·Mix's "Rain barely there"."""
    pick = single_winner(rows.values(), default, seed=seed, label="W2 level")
    lead = [o.option for o in rank(o for o in rows.values() if not o.out)[0]] if rows else []
    return {"rule": "section 8 Rain (level)", "rank": rank_doc(rank(rows.values())), "pick": pick,
            "rain_none_leads": RAIN_NONE in lead}


# ------------------------------------------------------------------------------------------------ loading
def built_by(runs: Path | None, rounds_of: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    """batch -> the lofi-shootout job ids whose pulled summary built it (`make pull` brings back
    runs/shootout-<job>/summary/): a batches.json, written by that job (its `run` is `shootout-<job id>`), that lists
    the batch with exactly the round ids of its manifest. Opaque round ids are random per batch, so a dry run's
    batches of the same names never match."""
    out: dict[str, list[str]] = {n: [] for n in rounds_of}
    if runs is None or not Path(runs).is_dir():
        return out
    for path in sorted(Path(runs).glob("shootout-*/summary/batches.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run = doc.get("run") if isinstance(doc, dict) else None
        job = run.removeprefix("shootout-") if isinstance(run, str) else ""
        if not job.isdigit() or not isinstance(doc.get("batches"), dict):
            continue
        for name, b in doc["batches"].items():
            ids = {r.get("id") for r in (b.get("rounds") or []) if isinstance(r, dict)}
            if name in out and ids == set(rounds_of[name]) and job not in out[name]:
                out[name].append(job)
    return out


def find_batches(inbox: Path) -> tuple[dict[str, Path], list[str]]:
    """batch name -> folder for every shootout batch at <inbox>/<date>/<batch>/, and the folders that did not load."""
    out: dict[str, Path] = {}
    problems: list[str] = []
    for man in sorted(Path(inbox).glob("*/*/manifest.json")):
        try:
            m = load_public(man.parent)
        except ManifestError as err:
            problems.append(f"{man.parent}: {err}")
            continue
        if m.mode != "shootout":
            continue
        if m.batch in out:
            raise TallyError(f"batch {m.batch} is in the inbox twice: {out[m.batch]} and {man.parent}")
        out[m.batch] = man.parent
    return out, problems


def micro_sessions(logs: Mapping[str, list[dict]]) -> dict[tuple, dict]:
    """(session, micro) -> its anchors, over every standing vote of every log: how many, how many caught (ui.audit's
    `anchor_caught`) and the weight of its votes. D-013: a micro-session that missed its anchor counts at half
    weight; one with several anchors missed them when it caught fewer than half (the audit's flag, CD-23 puts one
    anchor in each day)."""
    out: dict[tuple, dict] = {}
    for batch in sorted(logs):
        for v in active_votes(logs[batch]).values():
            hid = v.get("hidden") or {}
            if hid.get("kind") != "anchor":
                continue
            key = (v.get("session"), v.get("micro"))
            g = out.setdefault(key, {"session": key[0], "micro": key[1], "anchors": 0, "caught": 0, "batches": []})
            g["anchors"] += 1
            g["caught"] += int(anchor_caught(v.get("type"), v, hid.get("anchor")))
            if batch not in g["batches"]:
                g["batches"].append(batch)
    for g in out.values():
        g["missed"] = 2 * g["caught"] < g["anchors"]
        g["weight"] = 0.5 if g["missed"] else 1.0
    return out


def _vote(v: dict, infos: Sequence[dict], kind: str, weights: Mapping[tuple, dict], heard_min: int,
          problems: list[str], rid: str, heard: Mapping[tuple, str] | None = None) -> Vote:
    """One standing vote as the recount reads it. Its clips' heard bars are OR-ed over every attempt of the round in
    the batch's log (`heard`, ui.audit.heard_over_attempts: kept and undone votes and abandoned attempts; D-013 as
    amended), so bars heard before an undo or a reload count; `raw` is that merged copy, `own_bars` each clip's own."""
    own = {c.get("clip"): c.get("heard_bars") for c in v.get("clips") or []}
    v = with_merged_heard(v, heard if heard is not None else heard_over_attempts([v]))
    by_oid = {i["oid"]: i for i in infos}
    options = v.get("options") or []
    clips = []
    for i, c in enumerate(v.get("clips") or []):
        info = by_oid.get(c.get("oid")) or (infos[i] if i < len(infos) else {})
        opt = options[i] if i < len(options) else info.get("option")
        if opt != info.get("option"):
            problems.append(f"round {rid}: tile {c.get('pos')} is {opt!r} in the vote and {info.get('option')!r} in "
                            "the hidden manifest; the vote's own record counts")
        clips.append({"oid": c.get("oid"), "real": c.get("clip"), "pos": c.get("pos"), "option": opt,
                      "heard_bars": c.get("heard_bars"), "own_bars": own.get(c.get("clip")),
                      "heard": heard_bars(c) >= heard_min, "kit": info.get("kit")})
    hid = v.get("hidden") or {}
    if hid.get("kind", "none") != kind:
        problems.append(f"round {rid}: the vote says {hid.get('kind')!r} and the hidden manifest {kind!r}")
    key = (v.get("session"), v.get("micro"))
    return Vote(event_id=str(v.get("event_id")), ts=v.get("ts"), session=key[0], micro=key[1],
                weight=HALF if (weights.get(key) or {}).get("missed") else ONE, clips=tuple(clips),
                best=v.get("best"), worst=v.get("worst"), no_best=bool(v.get("no_best")),
                no_worst=bool(v.get("no_worst")), all_bad=bool(v.get("all_bad")),
                anchor=hid.get("anchor") if kind == "anchor" else None, raw=v)


@dataclass
class Batch:
    name: str
    dir: Path
    date: str
    day: int | None                   # the plan's day of this batch
    manifest: Any = field(repr=False)
    hidden: Any = field(repr=False)
    events: list[dict] = field(repr=False)
    log: bool
    rounds: list[Round]
    problems: list[str]

    @property
    def voted(self) -> int:
        return sum(rd.vote is not None for rd in self.rounds)

    @property
    def status(self) -> str:
        return "no_votes" if not self.voted else "complete" if self.voted == len(self.rounds) else "partial"


def load_tallied(name: str, folder: Path, votes_dir: Path, plan: Plan, weights: Mapping[tuple, dict],
                 heard_min: int) -> Batch:
    """One batch's manifests (no audio checks: the tally reads labels only) and its standing votes, each clip's
    heard bars OR-ed over every attempt of its round in the batch's log (D-013 as amended)."""
    m, h = load_batch(folder, check_files=False)
    path = Path(votes_dir) / f"{name}.jsonl"
    events, problems = read_events(path)
    act = active_votes(events)
    heard = heard_over_attempts(events)
    day = {n: b.day for b in plan.batches for n in b.names}.get(name)
    rounds = []
    for r in m.rounds:
        hr = h.rounds[r.id]
        infos = tuple({**hr.clips[c.id], "oid": c.id, "pos": c.pos} for c in r.clips)
        stages = {i.get("stage") for i in infos}
        stage = stages.pop() if len(stages) == 1 else None
        if not stage:
            problems.append(f"round {r.id}: its clips do not name one stage; tallied under its category")
            stage = hr.category or h.category or "?"
        base = plan.stages.get(stage)
        if (hr.kind != "repeat" and base is not None and day is not None and base.day != day
                and stage in plan.rerun_stages):
            stage += RERUN
        slots = {i.get("slot") for i in infos}
        v = act.get(r.id)
        rounds.append(Round(name, folder.parent.name, r.id, r.number, hr.kind, stage,
                            slots.pop() if len(slots) == 1 else None, hr.brief_id, hr.of, infos,
                            _vote(v, infos, hr.kind, weights, heard_min, problems, r.id, heard) if v else None))
    stray = sorted(set(act) - {r.id for r in m.rounds})
    if stray:
        problems.append(f"{len(stray)} vote(s) name rounds that are not in this manifest; left out")
    return Batch(name, folder, folder.parent.name, day, m, h, events, path.exists(), rounds, problems)


# ------------------------------------------------------------------------------------------------ figures
def _f(x) -> float | None:
    return None if x is None else float(x)


def _n(x) -> str:
    """A weighted count as text: 2, 1.5."""
    if x is None:
        return "-"
    x = float(x)
    return str(int(x)) if x == int(x) else f"{x:g}"


def row_doc(o: Row) -> dict:
    return {"option": o.option, "shown": o.shown, "heard_n": o.heard_n, "heard": _f(o.heard), "best": _f(o.best),
            "worst": _f(o.worst), "bw": _f(o.bw), "net": o.net,
            "audit": {"shown": o.a_shown, "best": o.a_best, "worst": o.a_worst, "net": o.net},
            "rounds": _f(o.rounds), "all_bad": _f(o.all_bad), "all_bad_n": o.all_bad_n,
            "unheard_marks": o.unheard_marks, "cannot_survive": o.out}


def flag_rules(plan: Plan, stage: str) -> list[tuple[str, float, str]]:
    """(hidden-manifest field, limit in LU, bus) for the rounds of a stage (section 3). A stage that names `match`
    (CD-53, Day 2 on) is checked on every bus it matches, under the fields the batch builder writes: `bus_lufs` for
    the stage's own bus (a vamp round's keys) and `<bus>_lufs` for each other one (R3a's sax, keys and bass, R3b's
    keys and bass, the vamps' bass)."""
    st = plan.stages.get(stage.removesuffix(RERUN))
    if st is None:
        return [("bus_lufs", MATCHED_LU, "bus")]
    if st.fills == BED_SLOT:      # W1 (section 4.5): every excerpt at delta 18, the bed measured in the render
        return [("bed_lu", MATCHED_LU, "bed")]
    if st.fills in UNMATCHED_SLOTS:                                  # W2: the level is the candidate
        return []
    limit = TIMING_LU if st.fills in TIMING_SLOTS else MATCHED_LU    # CD-26; instruments (CD-52), S2a, the vamps
    buses = list(st.match) or (["keys", "bass"] if st.fills is None else [st.bus])   # the vamps: keys and bass (CD-27)
    first = next(b for b in [*st.bus.split("+"), *MIX_ORDER] if b in buses)        # as batch._hidden_clip names it
    return [("bus_lufs", limit, first)] + [(f"{b}_lufs", limit, b) for b in MIX_ORDER if b in buses and b != first]


def round_levels(plan: Plan, rd: Round) -> tuple[list[dict], list[dict]]:
    """(flags, spreads) of one round from its hidden manifest's level fields. A Day-3 rain anchor round adds its sax
    (`sax_lufs`, at MATCHED_LU): its sour copy plays its own sax fader, matched to its round-mates' (2026-11-01
    review), where the music of a rain design round is one by construction."""
    flags, spreads = [], []
    rules = flag_rules(plan, rd.stage)
    st = plan.stages.get(rd.stage.removesuffix(RERUN))
    if rd.kind == "anchor" and st is not None and st.fills in UNMATCHED_SLOTS:
        rules = rules + [("sax_lufs", MATCHED_LU, "sax")]
    for name, limit, bus in rules:
        vals = [i.get(name) for i in rd.infos]
        where = {"batch": rd.batch, "round": rd.id, "number": rd.number, "kind": rd.kind, "stage": rd.stage,
                 "slot": rd.slot, "loop": rd.loop, "bus": bus, "field": name, "limit_lu": limit}
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vals):
            flags.append({**where, "what": "missing", "spread_lu": None})
            continue
        spread = round(max(vals) - min(vals), 4)
        spreads.append({**where, "spread_lu": spread})
        if spread > limit:
            flags.append({**where, "what": "spread", "spread_lu": spread})
    return flags, spreads


def levels_moved(paths: Sequence[Path]) -> list[dict]:
    """Section 3 "Drift between jobs": every trim or fader a job's levels tool moved by more than 0.1 dB."""
    out = []
    for p in paths:
        try:
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            raise TallyError(f"--levels {p}: {err}") from err
        if not isinstance(doc, dict) or "variants" not in doc:
            raise TallyError(f"--levels {p}: not a levels.json (no variants)")
        for vid in doc.get("moved") or []:
            rec = doc["variants"].get(vid) or {}
            out.append({"file": str(p), "variant": vid, "moved_db": rec.get("moved_db"), "prior": rec.get("prior"),
                        "limit_db": MOVE_DB})
        if doc.get("prior") is None:
            out.append({"file": str(p), "variant": None, "moved_db": None, "prior": None, "limit_db": MOVE_DB,
                        "note": "no --prior: nothing to compare"})
    return out


def _vote_day(ts) -> date | None:
    """The calendar day of a vote's timestamp, in its own offset (as `first_votes` reads it), or None."""
    try:
        return datetime.fromisoformat(ts).date()
    except (TypeError, ValueError):
        return None


def repeats_doc(rounds: Sequence[Round], logs: Mapping[str, list[dict]], heard_min: int, min_days: int) -> dict:
    """CD-47: each shootout repeat against its first showing (looked up in every log, as the audit does), on implied
    pairs, pooled with a Wilson 95 % interval. Both sides carry their heard bars OR-ed over every attempt of their
    round, each in its own batch's log (D-013 as amended; ui.audit.aggregate merges the same way). Repeats never
    enter the counts. D-013 pre-registers a repeat at least `min_days` days after its first showing
    (controls.repeats.min_days_after): a repeat whose standing vote falls fewer calendar days after its first
    showing's (each vote's date in its own offset) goes to `too_soon`, one without a readable date on either side to
    `undated`, and neither is scored (`days_after` on every row whose first showing was found)."""
    originals: dict[str, dict] = {}
    for b in sorted(logs):
        merged = heard_over_attempts(logs[b])
        originals.update({k: with_merged_heard(v, merged) for k, v in active_votes(logs[b]).items()})
    rows, agree, compared, pending = [], 0, 0, 0
    too_soon, undated = [], []
    for rd in rounds:
        if rd.kind != "repeat":
            continue
        if rd.vote is None:
            pending += 1
            continue
        first = originals.get(rd.of)
        row = {"batch": rd.batch, "round": rd.id, "stage": rd.stage, "slot": rd.slot, "of": rd.of,
               "of_batch": first.get("batch") if first else None, "found": first is not None}
        if first is not None:
            was, now = _vote_day(first.get("ts")), _vote_day(rd.vote.ts)
            row["days_after"] = (now - was).days if was and now else None
            if row["days_after"] is None:
                undated.append(row)
            elif row["days_after"] < min_days:
                too_soon.append(row)
            else:
                a, n = agreement("shootout", first, rd.vote.raw, heard_min)
                agree, compared = agree + a, compared + n
                row.update(agree=a, compared=n)
        rows.append(row)
    lo, hi = wilson(agree, compared)
    return {"rule": "CD-47", "rounds": len(rows) + pending, "voted": len(rows), "pending": pending,
            "found": sum(r["found"] for r in rows), "missing": sum(not r["found"] for r in rows),
            "min_days_after": min_days, "scored": sum("compared" in r for r in rows),
            "too_soon": [{k: r[k] for k in ("batch", "round", "stage", "of", "of_batch", "days_after")}
                         for r in too_soon],
            "undated": [{k: r[k] for k in ("batch", "round", "stage", "of", "of_batch")} for r in undated],
            "agree": agree, "compared": compared, "p": agree / compared if compared else None, "ci95": [lo, hi],
            "rows": rows, "caveat": REPEAT_CAVEAT}


def position_doc(audits: Mapping[str, dict]) -> dict:
    """The audits' position-of-best counts pooled by clip count over the shootout batches. Descriptive."""
    by_k: dict[int, list[int]] = {}
    for a in audits.values():
        for row in a.get("position_of_best") or []:
            pooled = by_k.setdefault(row["clips"], [0] * row["clips"])
            for i, n in enumerate(row["counts"]):
                pooled[i] += n
    rows = []
    for k, counts in sorted(by_k.items()):
        x2, p = chi2_uniform(counts)
        rows.append({"clips": k, "counts": counts, "chi2": x2, "df": k - 1, "p": p})
    return {"descriptive": True, "batches": sorted(audits), "rows": rows}


def first_votes(rounds: Sequence[Round], plan: Plan, serve: date | None = None) -> dict:
    """Section 2.5, for job 3's --repeats: per stage of the earlier days, the dates of its design rounds' standing
    votes (the logs' own timestamps, each in its own offset), first and last, and whether every one of its rounds
    was voted at least `min_days_after` days before Day 3 is served: `serve` (the CLI's --serve-date), or the plan's
    repeat day when none is given (`against` says which). A repeat may show any design round of the stage but the
    first served, so the stage is eligible only when its last vote is old enough and every design round has one."""
    rep = plan.controls["repeats"]
    when = serve or date.fromisoformat(plan.days[plan.repeat_day].date)
    dates: dict[str, list[date]] = {}
    design: dict[str, int] = {}
    for rd in rounds:
        if rd.kind != "none" or (plan.stages.get(rd.stage) and plan.stages[rd.stage].day >= plan.repeat_day):
            continue
        design[rd.stage] = design.get(rd.stage, 0) + 1
        if rd.vote is None or not rd.vote.ts:
            continue
        try:
            dates.setdefault(rd.stage, []).append(datetime.fromisoformat(rd.vote.ts).date())
        except ValueError:
            continue
    out = {}
    for s in sorted(design):
        ds = dates.get(s, [])
        first, last = (min(ds), max(ds)) if ds else (None, None)
        days = (when - last).days if last else None
        out[s] = {"first_vote": first.isoformat() if first else None, "last_vote": last.isoformat() if last else None,
                  "rounds": design[s], "voted": len(ds), "days_before": days,
                  "eligible": bool(ds) and len(ds) == design[s] and days >= rep["min_days_after"],
                  "against": when.isoformat(), "serve_date": serve is not None}
    return out


# ------------------------------------------------------------------------------------------------ the tally
def _stage_order(plan: Plan, stages: Iterable[str]) -> list[str]:
    names = set(stages)
    order = [s for s in plan.stages if s in names]
    order += [s + RERUN for s in plan.stages if s + RERUN in names]
    return order + sorted(names - set(order))


def stage_doc(plan: Plan, sid: str, rounds: Sequence[Round], loaded: Mapping[str, Batch]) -> dict:
    base = sid.removesuffix(RERUN)
    st = plan.stages.get(base)
    want = []
    if st is not None:
        rerun_days = {plan.rerun_day} if sid.endswith(RERUN) else {st.day}
        want = [n for b in plan.batches if b.day in rerun_days
                and (base in b.rounds or base in b.adds or sid.endswith(RERUN)) for n in b.names]
    missing = [n for n in want if n not in loaded]
    design = [rd for rd in rounds if rd.kind == "none"]
    voted = [rd for rd in design if rd.vote is not None]
    status = ("missing" if missing else "none" if not design else "no_votes" if not voted
              else "complete" if len(voted) == len(design) else "partial")
    rows = tally_rows(rounds)
    held = sorted({rd.batch for rd in rounds if rd.kind != "repeat"})
    waiting = [b for b in held if loaded[b].status != "complete"]
    doc = {"stage": sid, "category": st.category if st else None, "day": st.day if st else None,
           "batches": sorted({rd.batch for rd in rounds}), "missing_batches": missing, "status": status,
           # its rules wait for every round of its batches: an anchor there can halve a micro-session (D-013)
           "ready": status == "complete" and not waiting, "waiting_for": waiting,
           "design_rounds": len(design), "voted": len(voted),
           "anchor_rounds": sum(rd.kind == "anchor" for rd in rounds),
           "anchor_voted": sum(rd.kind == "anchor" and rd.vote is not None for rd in rounds),
           "repeat_rounds": sum(rd.kind == "repeat" for rd in rounds),
           "escape": {"no_best": sum(rd.vote.no_best for rd in voted),
                      "no_worst": sum(rd.vote.no_worst for rd in voted),
                      "all_bad": sum(rd.vote.all_bad for rd in voted)},
           "half_weight_votes": sum(rd.vote.weight != ONE for rd in voted),
           "unheard_marks": sum(o.unheard_marks for o in rows.values()),
           "options": [r for g in rank(rows.values()) for r in (row_doc(o) for o in g)] if voted else None,
           "places": rank_doc(rank(rows.values())) if voted else None, "moods": None}
    if voted and rows and all(o in plan.vamps for o in rows):     # the vamps rank within each mood (section 8)
        by_mood: dict[str, list[Row]] = {}
        for o, r in rows.items():
            by_mood.setdefault(plan.vamps[o].mood, []).append(r)
        doc["moods"] = {m: rank_doc(rank(rs)) for m, rs in sorted(by_mood.items())}
    return doc


def pooled_doc(rounds: Sequence[Round], keep: Iterable[str]) -> dict:
    """The options of a later stage (K2's participants, S1's finalists) over both stages' rounds."""
    keep = set(keep)
    rows = {o: r for o, r in tally_rows(rounds).items() if o in keep}
    groups = rank(rows.values())
    return {"options": [row_doc(o) for g in groups for o in g], "places": rank_doc(groups)}


def _apply(rules: dict, needs: list[dict], key: str, fn, *args, **kw) -> dict | None:
    """Run one rule; a RuleGap is recorded as raised and put to Shresth."""
    try:
        rules[key] = fn(*args, **kw)
    except RuleGap as err:
        rules[key] = {"raised": str(err)}
        needs.append({"stage": key, "what": str(err), "blocking": True})
        return None
    return rules[key]


def _note_pick(needs: list[dict], key: str, label: str, pick: Mapping) -> None:
    if pick.get("needs_word"):
        needs.append({"stage": key, "blocking": False,
                      "what": f"{label}: {pick['tied']} are fully tied and the default ({pick['default']}) is not "
                              f"among them (CD-39); provisional {pick['pick']}"})


def tally(votes_dir: Path, inbox: Path, *, plan: Plan, batches: Sequence[str] | None = None, heard_min: int = 1,
          toss_min: int = 1, budget_min: float = 60.0, levels: Sequence[Path] = (), today: date | None = None,
          runs: Path | None = None, day2_cuts: Sequence[str] | None = None,
          day2_cuts_by: str | None = None, serve_date: date | None = None, day3_cuts: Sequence[str] | None = None,
          day3_cuts_by: str | None = None) -> tuple[dict, dict[str, dict]]:
    """The tally of every shootout batch in `inbox` (or the named ones): (shootouts.json, {batch: <batch>.json}).
    `heard_min` and `toss_min` default to configs/rating.yaml's heard.min_bars and toss_min_bars, 1 (D-013 as
    amended; the CLI reads the config), over every attempt of a round. `runs` holds the pulled job summaries
    (runs/shootout-<job>/summary/): the job that built the Day-1 batches is job 2's --prior (section 3, drift).
    `day2_cuts` is Shresth's word on Day 2's cuts ([] for none), `day2_cuts_by` his words and their date: job 2's
    line then carries his cuts, not the cut list's (section 2.7). `serve_date` is the day Day 3 will be served (the CLI's
    --serve-date): job 3's --repeats name the stages whose rounds were all voted at least two days before it (section
    2.5), and job 3's ARGS wait for it. `day3_cuts` and `day3_cuts_by` are Shresth's word on Day 3's cuts, as for
    Day 2."""
    if day2_cuts is not None:
        day2_cuts = decided_cuts(plan, list(day2_cuts))
    elif day2_cuts_by is not None:
        raise TallyError("--day2-cuts-by goes with --day2-cuts")
    if day3_cuts is not None:
        day3_cuts = decided_cuts(plan, list(day3_cuts), day=3)
    elif day3_cuts_by is not None:
        raise TallyError("--day3-cuts-by goes with --day3-cuts")
    votes_dir, inbox = Path(votes_dir), Path(inbox)
    found, problems = find_batches(inbox)
    names = list(batches) if batches else sorted(found)
    unknown = [n for n in names if n not in found]
    if unknown:
        raise TallyError(f"--batch {unknown}: no shootout batch of that name under {inbox}/<date>/")
    logs = read_all(votes_dir)
    weights = micro_sessions(logs)
    loaded = {n: load_tallied(n, found[n], votes_dir, plan, weights, heard_min) for n in names}
    rounds = [rd for n in names for rd in loaded[n].rounds]
    for n in sorted(set(logs) - set(names)):
        if any(e.get("type") == "shootout" for e in logs[n]) and n not in found:
            problems.append(f"{votes_dir / (n + '.jsonl')}: shootout votes for a batch that is not in the inbox; "
                            "left out")

    batch_docs: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    flags, spreads = [], []
    for n in names:
        b = loaded[n]
        audit = None
        if b.voted:
            audit = aggregate(b.manifest, b.hidden, b.events, all_events=logs, heard_min_bars=heard_min,
                              toss_min_bars=toss_min, daily_budget_min=budget_min, today=today)
            audits[n] = audit
            _check_audit(n, audit, tally_rows(b.rounds))
        bf = []
        for rd in b.rounds:
            f, s = round_levels(plan, rd)
            bf += f
            spreads += s
        flags += bf
        batch_docs[n] = {
            "schema": BATCH_SCHEMA, "batch": n, "dir": str(b.dir), "date": b.date, "day": b.day,
            "category": b.hidden.category, "status": b.status, "log": b.log, "rounds": len(b.rounds),
            "voted": b.voted, "problems": b.problems, "audit": audit, "flags": bf,
            "votes": [_round_doc(rd) for rd in b.rounds]}

    by_stage: dict[str, list[Round]] = {}
    for rd in rounds:
        by_stage.setdefault(rd.stage, []).append(rd)
    stages = {sid: stage_doc(plan, sid, by_stage[sid], loaded) for sid in _stage_order(plan, by_stage)}
    rows = {sid: tally_rows(by_stage[sid]) for sid in stages}
    needs: list[dict] = []
    rules: dict[str, dict] = {}
    built = built_by(runs, {n: [rd.id for rd in loaded[n].rounds] for n in names})
    proposal = _day2(plan, stages, rows, by_stage, audits, {n: loaded[n].status for n in names}, rules, needs,
                     built=built, runs=runs, day2_cuts=day2_cuts, day2_cuts_by=day2_cuts_by)
    _later(plan, stages, rows, by_stage, rules, needs)
    eligibility = first_votes(rounds, plan, serve_date)
    proposal.update(_day3(plan, stages, by_stage, audits, {n: loaded[n].status for n in names}, rules, needs,
                          eligibility, built=built, runs=runs, serve_date=serve_date, day3_cuts=day3_cuts,
                          day3_cuts_by=day3_cuts_by))
    pooled = {}
    for name, a, b in POOLS:                             # BW over K1 + K2 (CD-40); combined S1 + final BW (CD-18)
        if (stages.get(a) or {}).get("voted") and (stages.get(b) or {}).get("voted"):
            pooled[name] = pooled_doc(by_stage[a] + by_stage[b], rows[b])
    largest: dict[str, dict] = {}
    for s in spreads:
        cur = largest.get(s["stage"] + "|" + s["bus"])
        if cur is None or s["spread_lu"] > cur["spread_lu"]:
            largest[s["stage"] + "|" + s["bus"]] = s
    doc = {
        "schema": SCHEMA, "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "votes": str(votes_dir), "inbox": str(inbox), "plan": {"path": str(plan.path), "sha256": plan.sha256},
        "heard_min_bars": heard_min, "problems": problems + [f"{n}: {p}" for n in names for p in loaded[n].problems],
        "batches": {n: {k: batch_docs[n][k] for k in ("date", "day", "status", "log", "rounds", "voted")}
                    | {"minutes": (audits[n]["time"]["minutes"] if n in audits else None), "jobs": built[n]}
                    for n in names},
        "runs": None if runs is None else str(runs),
        "stages": stages,
        "pooled": pooled,
        "micro_sessions": [{**g, "batches": sorted(g["batches"])} for _, g in sorted(
            weights.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0))],
        "repeats": repeats_doc(rounds, logs, heard_min, plan.controls["repeats"]["min_days_after"]),
        "position_of_best": position_doc(audits),
        "flags": flags,
        "largest_spread": sorted(largest.values(), key=lambda s: (s["stage"], s["bus"])),
        "levels_moved": levels_moved(levels),
        "first_votes": eligibility,
        "rules": rules,
        "proposal": proposal,
        "needs_word": needs,
    }
    for m in doc["levels_moved"]:
        if m["variant"] is not None:
            needs.append({"stage": "levels", "blocking": False,
                          "what": f"{m['variant']} moved {m['moved_db']} dB since the prior job (section 3, drift)"})
    return doc, batch_docs


def _check_audit(name: str, audit: dict, rows: Mapping[str, Row]) -> None:
    """The recount's audit counts must be ui.audit's own, option by option (a guard on this module)."""
    want = {o["option"]: (o["shown"], o["best"], o["worst"]) for o in audit["shootout"]["options"]}
    got = {o: (r.a_shown, r.a_best, r.a_worst) for o, r in rows.items() if r.a_shown or r.a_best or r.a_worst}
    if want != got:
        raise TallyError(f"{name}: the recount's audit counts {got} differ from ui.audit.aggregate's {want}")


def _round_doc(rd: Round) -> dict:
    """A round as the private batch file keeps it: its options by tile and his picks."""
    v = rd.vote
    doc = {"round": rd.id, "number": rd.number, "kind": rd.kind, "stage": rd.stage, "slot": rd.slot, "loop": rd.loop,
           "of": rd.of, "voted": v is not None}
    if v is None:
        return doc
    opt = {c["real"]: c["option"] for c in v.clips}
    doc.update(event_id=v.event_id, ts=v.ts, session=v.session, micro=v.micro, weight=float(v.weight),
               best=opt.get(v.best), worst=opt.get(v.worst), no_best=v.no_best, no_worst=v.no_worst,
               all_bad=v.all_bad, anchor=opt.get(v.anchor) if v.anchor else None,
               clips=[{"pos": c["pos"], "option": c["option"], "heard_bars": c["heard_bars"], "heard": c["heard"],
                       **({"own_bars": c["own_bars"]} if c["own_bars"] != c["heard_bars"] else {})}
                      for c in v.clips])
    return doc


def _defaults(plan: Plan) -> dict[str, str]:
    """The current default of each band slot: the Day-1 band (the dev palette's instruments) and the plan's stage
    defaults (CD-39)."""
    out = {BUS_SLOT[bus]: inst for bus, inst in plan.days[1].band.items()}
    for st in plan.stages.values():
        if st.default is not None and st.fills:
            out[st.fills] = st.default
    return out


def _day2(plan: Plan, stages: Mapping[str, dict], rows: Mapping[str, dict], by_stage: Mapping[str, list[Round]],
          audits: Mapping[str, dict], batches: Mapping[str, str], rules: dict, needs: list[dict], *,
          built: Mapping[str, Sequence[str]] | None = None, runs: Path | None = None,
          day2_cuts: Sequence[str] | None = None, day2_cuts_by: str | None = None) -> dict:
    """Day 1's rules and job 2's ARGS (section 7.7): --leaders, --prior, the job that built the Day-1 batches
    (`built`), whose drift keys job 2 compares with its own (section 3; jobs/shootout.sbatch's Day-2 form), --add R2r
    when the bass is undecided (CD-54), and --cut: the cut list's, on the minutes of the Day-2 plan those leaders and
    adds give, or Shresth's word on the cuts (`day2_cuts`, checked by `decided_cuts`), which then decides the S1
    rule's final, the band check and the ARGS, the list's cuts staying in rules["cut"] as advice. All once every
    round of the Day-1 batches has a vote (the anchor in the last one can halve a micro-session that began in an
    earlier one, D-013)."""
    status = {s: (stages[s]["status"] if s in stages else "missing") for s in DAY1_STAGES}
    day1 = [n for b in plan.batches if b.day == 1 for n in b.names]
    unfinished = [n for n in day1 if batches.get(n) != "complete"]
    if any(v != "complete" for v in status.values()) or unfinished:
        return {"day2": {"status": "incomplete", "stages": status, "waiting_for": unfinished}}
    dflt = _defaults(plan)
    keys = _apply(rules, needs, "K1", keys_to_k2, rows["K1"], vote_seed(by_stage["K1"]), default=dflt.get("keys"))
    kits = _apply(rules, needs, "R1", kits_after_r1, rows["R1"], vote_seed(by_stage["R1"]), default=dflt.get("kit"))
    bass = relisten = None
    if kits is not None:
        bass = _apply(rules, needs, "R2", bass_pairings, by_stage["R2"], rows["R2"], kits["survivors"],
                      vote_seed(by_stage["R2"]), default=dflt.get("bass"))
    if bass is not None:                    # CD-54: an undecided bass adds R2r; the default stays meanwhile
        relisten = bass["relisten"] = relisten_bass(by_stage["R2"], bass, kits, plan, default=dflt.get("bass"))
    adds = [RELISTEN] if relisten and relisten["adds"] else []
    counts = {"keys": len(keys["k2"])} if keys else {}
    try:                                    # the final's size (2 or 3), for the minutes of the plan job 2 draws
        counts["sax"] = len(sax_to_final(rows["S1"], by_stage["S1"], vote_seed(by_stage["S1"]),
                                         default=dflt.get("sax"))["sax"])
    except RuleGap:
        pass
    cut = day2_cut(plan, sum(audits[n]["time"]["minutes"] for n in day1 if n in audits), counts=counts, adds=adds,
                   decided=day2_cuts, by=day2_cuts_by)
    rules["cut"] = cut
    cuts = cut["cuts"] if cut["decided"] is None else cut["decided"]["cuts"]      # his word overrides the list
    if cut["decided"] is None and not cut["fits"]:
        needs.append({"stage": "cut", "blocking": False,
                      "what": f"Day 2 does not fit even with every cut (r = {cut['r']}): {cut['day2_minutes']} min "
                              f"for Day 2, {cut['days12_minutes']} for Days 1-2 (section 2.7)"})
    sax = _apply(rules, needs, "S1", sax_to_final, rows["S1"], by_stage["S1"], vote_seed(by_stage["S1"]),
                 default=dflt.get("sax"), final="final" not in cuts)
    if keys:
        _note_pick(needs, "K1", "the K1 leader (the Day-2 band's keys)", keys["leader"])
    if kits:
        _note_pick(needs, "R1", "the R1 leader (the Day-2 band's kit)", kits["leader"])
    if adds:
        rounds = "; ".join(f"`{k}` in {'+'.join(relisten['rounds'][k])}" for k in relisten["kits"])
        needs.append({"stage": "R2", "blocking": False,
                      "what": f"the bass is undecided ({'; '.join(relisten['why'])}): Day 2 adds R2r, the 3-round bass "
                              f"re-listen (--add R2r: {rounds}), and the Day-2 band keeps the default bass, "
                              f"{relisten['bass']}, meanwhile (CD-54); R2r decides it after Day 2 (CD-43)"})
        if relisten["left_out"]:
            needs.append({"stage": "R2", "blocking": False,
                          "what": f"{relisten['left_out']} survive R1 but play no R2r round (it plays "
                                  f"{len(relisten['kits'])} kits); after Day 2 they take the pooled R2r leader "
                                  "(CD-54)"})
    elif bass and any(p["from"] == "the pooled R2 leader" for p in bass["pairings"].values()):
        _note_pick(needs, "R2", "the pooled R2 leader", bass["pooled_leader"])
    if sax:
        _note_pick(needs, "S1", "the S1 leader (S2 runs on it)", sax["leader"])
        for why in sax["no_free_sax"]:
            needs.append({"stage": "S1", "blocking": True,
                          "what": f"no free sax passes: {why}; D-012's fallback (the lead becomes Rhodes) is Shresth's "
                                  "call (section 8 Sax)"})
    if keys is None or kits is None or bass is None or sax is None:
        raised = [rules[k]["raised"] for k in DAY1_STAGES if "raised" in rules.get(k, {})]
        return {"day2": {"status": "blocked", "stages": status, "raised": raised}}
    lead_kit = kits["leader"]["pick"]
    if adds:                                # CD-54: the surviving kits R2r plays, leader first; the default bass
        kit_ids, bass_ids = tuple(relisten["kits"]), (relisten["bass"],)
    else:
        kit_ids, bass_ids = (lead_kit,), (bass["pairings"][lead_kit]["bass"],)
    leaders = {"keys": tuple(keys["k2"]), "kit": kit_ids, "bass": bass_ids, "sax": tuple(sax["sax"])}
    text = leaders_string(leaders)
    parsed = parse_leaders(text)
    if parsed != leaders:
        raise TallyError(f"the --leaders string {text!r} does not read back as {leaders}")
    try:
        band = day_band(plan, "day2", parsed, cuts, adds)
    except (DrawError, PlanError) as err:
        raise TallyError(f"the proposal {text!r} fails the Day-2 band check: {err}") from err
    built = built or {}
    jobs = sorted({j for n in day1 for j in built.get(n) or []}, key=int)
    unbuilt = [n for n in day1 if not built.get(n)]
    if unbuilt:                 # job 2 needs job 1 as --prior: its drift keys and each instrument's move (section 3)
        jobs = []
        where = f"under {runs}" if runs is not None else "(no runs folder given)"
        needs.append({"stage": "prior", "blocking": True,
                      "what": f"job 2's --prior: no pulled summary {where} built {unbuilt} "
                              "(runs/shootout-<job>/summary/batches.json, from make pull); job 2 compares its drift "
                              "keys with job 1's (section 3), so add --prior <job 1> before submitting"})
    elif len(jobs) > 1:
        needs.append({"stage": "prior", "blocking": False,
                      "what": f"the Day-1 batches come from {len(jobs)} jobs ({'+'.join(jobs)}); job 2 compares its "
                              "drift keys with the first one's levels.json (section 3)"})
    args = (f"--stage day2 --leaders {text}" + (f" --prior {'+'.join(jobs)}" if jobs else "")
            + (f" --add {'+'.join(adds)}" if adds else "")
            + (f" --cut {'+'.join(cuts)}" if cuts else ""))                    # jobs/shootout.sbatch's order
    blocking = [n["what"] for n in needs if n["blocking"] and n["stage"] in (*DAY1_STAGES, "prior")]
    return {"day2": {"status": "ready", "stages": status, "leaders": text, "cut": cuts, "add": adds,
                     "prior": jobs, "args": args, "blocking": blocking,
                     "submit": f"make submit JOB=jobs/shootout.sbatch ARGS='{args}'",
                     "check": {"parse_leaders": {k: list(v) for k, v in parsed.items()}, "day_band": band},
                     "kits": kits["survivors"],
                     "pairings": {k: p["bass"] for k, p in bass["pairings"].items()},
                     "relisten": relisten if adds else None}}


# ------------------------------------------------------------------------------------------------ after Day 2
def job_leaders(runs: Path | None, job: str) -> dict | None:
    """A pulled job's summary/leaders.json (everything it was given: --leaders, --add, --cut, --prior), or None."""
    if runs is None:
        return None
    path = Path(runs) / f"shootout-{job}" / "summary" / "leaders.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != LEADERS_SCHEMA or not isinstance(doc.get("leaders"), dict):
        return None
    return {**doc, "path": str(path)}


def repeat_minutes(plan: Plan, by_stage: Mapping[str, Sequence[Round]], eligible: Sequence[str]) -> dict:
    """The plan estimate of Day 3's repeats (section 2.7, "drawn"): per repeat group with an eligible stage, the mean
    of `round_seconds` over the design rounds a repeat may show (every round of its eligible stages but the first
    one served of each, CD-24), since the draw picks one at job time."""
    out = {}
    for group, members in plan.repeat_groups.items():
        pool = []
        for s in members:
            if s not in eligible:
                continue
            design = sorted((rd for rd in by_stage.get(s, []) if rd.kind == "none"), key=lambda rd: (rd.batch, rd.number))
            pool += [round_seconds(len(rd.infos), plan.loop_named(rd.loop).brief.bpm) for rd in design[1:]]
        if pool:
            out[group] = sum(pool) / len(pool)
    return {"groups": {g: round(s / 60.0, 2) for g, s in out.items()}, "seconds": sum(out.values()),
            "minutes": round(sum(out.values()) / 60.0, 1)}


def day3_cut(plan: Plan, minutes12: float, plan12: float, repeats_s: float, rerun_s: float, *,
             decided: Sequence[str] | None = None, by: str | None = None) -> dict:
    """Section 2.7, Day 3: cut only if r x its plan would pass 30 minutes, first W1 down to 2 rounds (`w1-two`), then
    W2 (`w2-two`). r is Days 1-2's audit minutes over their plans (Day 1's batches, Day 2's as job 2 drew it: a
    choice the spec left open, since its r was read after Day 1); Day 3's plan is the drawn plan's estimate of the
    rain rounds and the anchor (`draw.day_minutes`), the repeats' (`repeat_minutes`) and an S2 re-run's rounds. Each
    cut saves the drop it makes in that estimate. `decided` is Shresth's word on Day 3's cuts, as for Day 2."""
    r = minutes12 / plan12 if plan12 else 0.0

    def p3(cuts: Sequence[str]) -> float:
        return (day_minutes(plan, 3, cuts=cuts)["seconds"] + repeats_s + rerun_s) / 60.0

    base = p3([])
    cuts, m = [], base
    for c in sorted((c for c in plan.cuts if c.day == 3), key=lambda c: c.order):
        if r * m <= DAY3_MIN:
            break
        cuts.append(c.id)
        m = p3(cuts)
    doc = {"rule": "section 2.7, Day 3 (CD-50)", "days12_minutes": round(minutes12, 1), "days12_plan": round(plan12, 1),
           "r": round(r, 4), "day3_plan": round(base, 1), "repeats_minutes": round(repeats_s / 60.0, 1),
           "rerun_minutes": round(rerun_s / 60.0, 1), "cuts": cuts, "saves_min": round(base - m, 1),
           "day3_minutes": round(r * m, 1), "fits": r * m <= DAY3_MIN, "decided": None}
    if decided is not None:
        md = p3(list(decided))
        doc["decided"] = {"cuts": list(decided), "by": by or WORD, "saves_min": round(base - md, 1),
                          "day3_minutes": round(r * md, 1), "fits": r * md <= DAY3_MIN}
    return doc


def _day3(plan: Plan, stages: Mapping[str, dict], by_stage: Mapping[str, list[Round]], audits: Mapping[str, dict],
          batches: Mapping[str, str], rules: dict, needs: list[dict], eligibility: Mapping[str, dict], *,
          built: Mapping[str, Sequence[str]] | None, runs: Path | None, serve_date: date | None,
          day3_cuts: Sequence[str] | None, day3_cuts_by: str | None) -> dict:
    """Job 3's ARGS (section 7.7) from the Day-1 and Day-2 votes and job 2's pulled leaders.json, once every round of
    the Day-1 and Day-2 batches has a vote and the after-Day-2 rules have run:
    - `--leaders`, one id per slot and every variant slot named (section 4.5 "the Day-2 picks everywhere"): the keys
      leader over K1 + K2 (CD-40); the kit job 2 ran as the R1 leader (its leaders.json, Shresth's confirmed word)
      with that kit's bass (from its R2r rounds when job 2 added R2r, CD-54; else the R2 pairing job 2 played); the
      sax final's winner (job 2's S1 leader when it cut the final); the S2a warmth, the S2b timing, the R3a swing
      (advisory: the default unless Shresth names another, CD-13) and the R3b humanize;
    - `--prior`: the jobs whose pulled summaries built the Day-1 and Day-2 batches;
    - `--repeats`: the stages every design round of which was voted at least two days before `serve_date` (section
      2.5, `first_votes`), in plan order; a group with none gets no repeat;
    - `--serve-date`: `serve_date`, which job 3 writes into every Day-3 hidden manifest (`serve_from`), so `make
      rate` serves no Day-3 batch before it (job 3 refuses --repeats without it);
    - `--rerun-s2`: the final's winner when it is not the sax job 2 ran S2 on (its first sax id, section 4.4);
    - `--cut`: `day3_cut`'s, or Shresth's word (`day3_cuts`).
    The status is "ready" only with a serving date and nothing blocking."""
    day12 = [n for b in plan.batches if b.day in (1, 2) for n in b.names]
    day2 = [n for b in plan.batches if b.day == 2 for n in b.names]
    unfinished = [n for n in day12 if batches.get(n) != "complete"]
    if unfinished:
        return {"day3": {"status": "incomplete", "waiting_for": unfinished}}
    built = built or {}
    jobs1 = sorted({j for n in day12 if n not in day2 for j in built.get(n) or []}, key=int)
    jobs2 = sorted({j for n in day2 for j in built.get(n) or []}, key=int)
    where = f"under {runs}" if runs is not None else "(no runs folder given)"
    if len(jobs2) != 1 or len(jobs1) != 1:
        what = ("job 3's ARGS: no pulled summary " + where + " built the Day-1 and the Day-2 batches, one job each "
                f"(found {jobs1 or 'none'} and {jobs2 or 'none'}; runs/shootout-<job>/summary/batches.json, from make "
                "pull): job 3 needs both as --prior, and job 2's leaders.json")
        needs.append({"stage": "day3", "blocking": True, "what": what})
        return {"day3": {"status": "blocked", "raised": [what]}}
    j2 = job_leaders(runs, jobs2[0])
    if j2 is None or j2.get("stage") != "day2":
        what = f"job 3's ARGS: runs/shootout-{jobs2[0]}/summary/leaders.json (what job 2 ran) is missing or not job 2's"
        needs.append({"stage": "day3", "blocking": True, "what": what})
        return {"day3": {"status": "blocked", "raised": [what]}}
    ran = {k: tuple(v) for k, v in j2["leaders"].items()}
    adds, cuts2 = list(j2.get("add") or []), list(j2.get("cut") or [])
    p2 = apply_cuts(apply_adds(plan, adds), cuts2)
    want = [s.id for s in p2.stages_of(2)]
    waiting = [s for s in want if not (stages.get(s) or {}).get("ready")]
    if waiting:
        return {"day3": {"status": "incomplete", "waiting_for_stages": waiting}}
    raised = [f"{k}: {rules[k]['raised']}" for k in (*DAY1_STAGES, *want) if "raised" in (rules.get(k) or {})]
    if raised:
        return {"day3": {"status": "blocked", "raised": raised}}
    notes, blocking = [], []
    kit = ran["kit"][0]
    r1 = ((rules.get("R1") or {}).get("leader") or {}).get("pick")
    if r1 is not None and r1 != kit:
        notes.append(f"job 2 ran {kit} as the R1 leader, and the tally reads {r1}: Day 3 plays job 2's (Shresth's "
                     "confirmed band)")
    if RELISTEN in adds:
        pairings = {k: p["bass"] for k, p in rules[RELISTEN]["pairings"].items()}
        bass = pairings[kit]
    else:
        pairings = {k: p["bass"] for k, p in ((rules.get("R2") or {}).get("pairings") or {}).items()}
        bass = ran["bass"][0]
    s2_on = ran["sax"][0]
    if "S1f" in want:
        f = rules["S1f"]
        sax = f["winner"] or s2_on
        if f["winner"] is None:
            blocking.append(f"the sax final is tied ({f['tied']}): Shresth's word names Day 3's sax (section 8 Sax)")
    else:
        sax = s2_on
    rerun = sax if sax != s2_on else None
    leaders = {"keys": (rules["K2"]["leader"]["pick"],), "kit": (kit,), "bass": (bass,), "sax": (sax,),
               "swing": (rules["R3a"]["pick"],), "humanize": (rules["R3b"]["pick"]["pick"],),
               "sax_variant": (rules["S2a"]["pick"]["pick"],), "timing": (rules["S2b"]["pick"]["pick"],)}
    if rerun:
        notes.append(f"the final picks {sax}, not {s2_on}, on whom job 2 ran S2: S2 re-runs on Day 3 (--rerun-s2 "
                     f"{sax}), and the rain rounds play the S2 picks of Day 2, which render the same night")
    text = leaders_string(leaders)
    parsed = parse_leaders(text)
    if parsed != leaders:
        raise TallyError(f"the --leaders string {text!r} does not read back as {leaders}")
    eligible = [s for s in plan.stages if (eligibility.get(s) or {}).get("eligible")]
    no_repeat = [g for g, members in plan.repeat_groups.items() if not any(s in eligible for s in members)]
    repeats = repeat_minutes(plan, by_stage, eligible)
    rerun_s = (sum(round_seconds(len(sl.tiles), plan.loops[sl.loop].brief.bpm) for s in plan.rerun_stages
                   for sl in plan.stages[s].slots) if rerun else 0.0)
    minutes12 = sum(audits[n]["time"]["minutes"] for n in day12 if n in audits)
    plan12 = (sum(b.est_min_plan for b in plan.batches if b.day == 1)
              + day_minutes(plan, 2, {"keys": len(ran["keys"]), "sax": len(ran["sax"])}, cuts2, adds)["seconds"] / 60.0)
    cut = day3_cut(plan, minutes12, plan12, repeats["seconds"], rerun_s, decided=day3_cuts, by=day3_cuts_by)
    rules["day3_cut"] = cut
    cuts3 = cut["cuts"] if cut["decided"] is None else cut["decided"]["cuts"]
    if cut["decided"] is None and not cut["fits"]:
        notes.append(f"Day 3 does not fit even with every cut (r = {cut['r']}): {cut['day3_minutes']} min (section 2.7)")
    try:
        band = day_band(plan, "day3", parsed, cuts3)
    except (DrawError, PlanError) as err:
        raise TallyError(f"the proposal {text!r} fails the Day-3 band check: {err}") from err
    prior = sorted({*jobs1, *jobs2}, key=int)
    if serve_date is None:
        blocking.append("job 3's --repeats need the date Day 3 will be served (--serve-date): a repeat comes at least "
                        "two days after its round was voted (section 2.5)")
    for g in no_repeat:
        notes.append(f"no stage of the repeat group {g} was voted at least 2 days before the serving date: it gets no "
                     "repeat (section 2.5)")
    for n in notes:
        needs.append({"stage": "day3", "blocking": False, "what": n})
    for b in blocking:
        needs.append({"stage": "day3", "blocking": True, "what": b})
    args = (f"--stage day3 --leaders {text} --prior {'+'.join(prior)}"
            + (f" --repeats {'+'.join(eligible)}" if eligible else "")
            + (f" --serve-date {serve_date.isoformat()}" if serve_date else "")      # the batches' serve_from
            + (f" --rerun-s2 {rerun}" if rerun else "") + (f" --cut {'+'.join(cuts3)}" if cuts3 else ""))
    status = "ready" if not blocking else "needs_serve_date" if serve_date is None and len(blocking) == 1 else "blocked"
    return {"day3": {"status": status, "leaders": text, "prior": prior, "repeats": eligible, "no_repeat": no_repeat,
                     "serve_date": serve_date.isoformat() if serve_date else None, "rerun_s2": rerun, "cut": cuts3,
                     "args": args, "blocking": blocking, "notes": notes,
                     "submit": f"make submit JOB=jobs/shootout.sbatch ARGS='{args}'",
                     "job2": {"job": jobs2[0], "leaders": {k: list(v) for k, v in ran.items()}, "add": adds,
                              "cut": cuts2, "file": j2["path"]},
                     "check": {"parse_leaders": {k: list(v) for k, v in parsed.items()}, "day_band": band},
                     "pairings": pairings, "keys": list(rules["K2"]["survivors"]),
                     "repeat_minutes": repeats["minutes"]}}


def _later(plan: Plan, stages: Mapping[str, dict], rows: Mapping[str, dict], by_stage: Mapping[str, list[Round]],
           rules: dict, needs: list[dict]) -> None:
    """The later days' rules, each once every round of its stages' batches has a vote."""
    dflt = _defaults(plan)
    done = {s for s, d in stages.items() if d["ready"]}
    if {"K1", "K2"} <= done:
        pooled = tally_rows(by_stage["K1"] + by_stage["K2"])
        k = _apply(rules, needs, "K2", keys_survivors, pooled, sorted(rows["K2"]),
                   vote_seed(by_stage["K1"] + by_stage["K2"]), default=dflt.get("keys"))
        if k:
            _note_pick(needs, "K2", "the keys leader over K1 + K2 (Day 3's keys)", k["leader"])
    if {"S1", "S1f"} <= done:
        s1 = rules.get("S1") or {}
        lead = (s1.get("leader") or {}).get("pick")
        f = _apply(rules, needs, "S1f", sax_final, rows["S1f"], tally_rows(by_stage["S1"] + by_stage["S1f"]),
                   sorted(rows["S1f"]), s1_leader=lead)
        if f and f["needs_word"]:
            needs.append({"stage": "S1f", "blocking": True,
                          "what": f"the sax final is tied ({f['tied']}) on bests and on combined S1 + final BW: "
                                  "Shresth's word (section 8 Sax)"})
        if f and f["rerun_s2"]:
            needs.append({"stage": "S1f", "blocking": False,
                          "what": f"the final picks {f['rerun_s2']}, not the S1 leader {lead}: S2 re-runs on Day 3 "
                                  f"(--rerun-s2 {f['rerun_s2']}), if job 2 ran S2 on {lead}"})
    for sid, key in (("S2a", "S2a"), ("S2a" + RERUN, "S2a" + RERUN)):
        if sid in done:
            w = _apply(rules, needs, key, warmth, rows[sid], vote_seed(by_stage[sid]), default=dflt.get("sax_variant"))
            if w:
                _note_pick(needs, key, "warmth", w["pick"])
                if w["control_top"]:
                    needs.append({"stage": key, "blocking": False,
                                  "what": f"{CONTROL} has the top BW in S2a: does expression stay on? (section 8)"})
    for sid in ("S2b", "S2b" + RERUN):
        if sid in done:
            t = _apply(rules, needs, sid, timing, rows[sid], vote_seed(by_stage[sid]), default=dflt.get("timing"))
            if t:
                _note_pick(needs, sid, "sax timing", t["pick"])
    if "R3a" in done:
        s = _apply(rules, needs, "R3a", swing, rows["R3a"], default=dflt.get("swing"))
        if s:
            needs.append({"stage": "R3a", "blocking": False,
                          "what": f"swing stays {s['pick']} unless Shresth names another after the R3a tally (BW "
                                  f"leader {s['bw_leader']}; {s['note']}; CD-13)"})
    if "R3b" in done:
        h = _apply(rules, needs, "R3b", humanize, rows["R3b"], vote_seed(by_stage["R3b"]),
                   default=dflt.get("humanize"))
        if h:
            _note_pick(needs, "R3b", "humanize", h["pick"])
    if RELISTEN in done:                    # CD-54: R2r decides each surviving kit's bass (CD-43)
        # the kits R2r played, in slot order (the leader first), then any other R1 survivor (it plays no R2r round)
        kits = list(dict.fromkeys(rd.kit for rd in sorted(by_stage[RELISTEN], key=lambda r: r.slot or "")
                                  if rd.kind == "none" and rd.kit))
        r1 = rules.get("R1") or {}
        if r1.get("survivors"):
            lead = r1["leader"]["pick"]
            kits += [k for k in [lead, *r1["survivors"]] if k not in kits]
        b = _apply(rules, needs, RELISTEN, bass_after_relisten, by_stage[RELISTEN], kits, default=dflt.get("bass"))
        if b:
            pooled_used = any(p["from"] == "the pooled R2r leader" for p in b["pairings"].values())
            if pooled_used:
                _note_pick(needs, RELISTEN, "the pooled R2r leader", b["pooled_leader"])
            for kit, p in b["pairings"].items():
                if p["from"] == "its own R2r rounds":
                    _note_pick(needs, RELISTEN, f"the bass of {kit} over its R2r rounds", p["pick"])
    if "V" in done:
        v = _apply(rules, needs, "V", vamp_ranks, rows["V"], {k: x.mood for k, x in plan.vamps.items()})
        if v and v["borrowed"]["met"]:
            needs.append({"stage": "V", "blocking": False,
                          "what": "borrowed chords: float ranks 4th in wistful and plagal 1st in melancholy; nothing "
                                  "leaves chords.numerals without Shresth's words (section 8 Vamps, CD-45, section 11 "
                                  "question 2)"})
    if "W1" in done:
        w1 = _apply(rules, needs, "W1", rain_excerpt, rows["W1"], vote_seed(by_stage["W1"]), default=dflt.get("bed"))
        if w1:
            _note_pick(needs, "W1", "the rain excerpt", w1["pick"])
    if "W2" in done:
        w2 = _apply(rules, needs, "W2", rain_level, rows["W2"], vote_seed(by_stage["W2"]), default=dflt.get("rain"))
        if w2:
            _note_pick(needs, "W2", "the rain level", w2["pick"])
            if w2["rain_none_leads"]:
                needs.append({"stage": "W2", "blocking": False,
                              "what": "rain_none leads W2: no bed changes T·Mix's \"Rain barely there\", so it is "
                                      "Shresth's call (section 8 Rain)"})


# ------------------------------------------------------------------------------------------------ the report
def _bw(x) -> str:
    if x is None:
        return "n/a"
    s = f"{x:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s


def _signed(x: int) -> str:
    return f"{x:+d}" if x else "0"


def _table(doc: dict, places: Sequence[dict] | None = None) -> list[str]:
    """A stage's (or pool's) table in CD-39 order; with `places`, only those options, placed as given."""
    keep = {p["option"] for p in places} if places is not None else None
    place = {p["option"]: f"{p['place']}{'=' if p['tied'] else ''}" for p in (places or doc["places"] or [])}
    lines = ["| place | option | shown | heard | best | worst | BW | net | all-bad |",
             "|---|---|---|---|---|---|---|---|---|"]
    for o in doc["options"] or []:
        if keep is not None and o["option"] not in keep:
            continue
        mark = " (cannot survive, CD-44)" if o["cannot_survive"] else ""
        lines.append(f"| {place.get(o['option'], '')} | `{o['option']}` | {o['shown']} | {_n(o['heard'])} | "
                     f"{_n(o['best'])} | {_n(o['worst'])} | {_bw(o['bw'])} | {_signed(o['net'])} | "
                     f"{_n(o['all_bad'])} of {_n(o['rounds'])}{mark} |")
    return lines


def _pick_text(p: Mapping) -> str:
    t = f"`{p['pick']}` ({p['by']})"
    if p.get("draw"):
        t += f"; seeded draw {p['draw']['seed']} among {p['draw']['among']}"
    return t


def shown_path(path) -> str:
    """A path as the report shows it: relative to the repo when it is inside it."""
    p = Path(path).expanduser().resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def report_md(doc: dict, out_dir: str | None = None) -> str:
    """The tally as markdown for gates/GATE_palette.md (section 8 "What is recorded"). Per option, never per tile."""
    n = doc["heard_min_bars"]
    L = [f"# Shootout tally ({doc['generated']})", "",
         f"Made by `python -m lofi_rl.shootout.tally` from the vote logs in `{shown_path(doc['votes'])}` and the "
         f"hidden manifests in `{shown_path(doc['inbox'])}`; the rules are docs/SHOOTOUTS.md section 8. Heard = "
         f"{n} bar{'' if n == 1 else 's'} or more, over every attempt of a round (D-013 as amended). Counts only: "
         "nothing here says how anything sounds."]
    if out_dir:
        files = [f"`{out_dir}/{n}.json`" for n in doc["batches"]] + [f"`{out_dir}/shootouts.json`"]
        L.append(f"The numbers come from {', '.join(files)} (private).")
    L += ["", "## Batches", "", "| batch | date | rounds | voted | status | minutes |", "|---|---|---|---|---|---|"]
    words = {"no_votes": "no votes yet", "partial": "partial", "complete": "complete"}
    for n, b in doc["batches"].items():
        L.append(f"| `{n}` | {b['date']} | {b['rounds']} | {b['voted']} | {words[b['status']]} | "
                 f"{_n(b['minutes']) if b['minutes'] is not None else '-'} |")
    L += ["", "## Stages", "",
          "Columns: shown = appearances in design rounds (anchor rounds dropped, repeats left out); heard = those "
          "heard for the minimum bars over every attempt of their round, weighted (a micro-session that missed its "
          "anchor counts half, D-013); best and worst = marks on heard clips, weighted; BW = (best - worst) / heard; "
          "net = the audit's best - worst over "
          "every shown clip; all-bad = rounds marked G of the rounds shown, weighted (CD-44: half or more cannot "
          "survive). Places follow CD-39 (BW, then more bests, then fewer worsts); `=` marks a full tie."]
    for sid, s in doc["stages"].items():
        L += ["", f"### {sid} ({s['category'] or 'not in the plan'}; {', '.join(s['batches'])})", ""]
        if s["status"] == "missing":
            L.append(f"Batch(es) not read: {', '.join(s['missing_batches'])}.")
            if not s["voted"]:
                continue
        if s["status"] == "no_votes":
            L.append(f"No votes yet ({s['design_rounds']} design rounds).")
            continue
        if s["status"] == "none":
            L.append("No design round in the batches read.")
            continue
        if s["status"] == "partial":
            L.append(f"Partial: {s['voted']} of {s['design_rounds']} design rounds voted; no rule is applied.")
        elif s["waiting_for"]:
            L.append(f"{s['voted']} of {s['design_rounds']} design rounds voted; its rules wait for the rest of "
                     f"{', '.join(s['waiting_for'])} (an anchor there can halve a micro-session, D-013).")
        else:
            L.append(f"{s['voted']} of {s['design_rounds']} design rounds voted.")
        if s.get("moods"):
            for mood, places in s["moods"].items():
                L += ["", f"{mood}:", ""] + _table(s, places)
        else:
            L += [""] + _table(s)
        e = s["escape"]
        L += ["", f"Escape hatches: no clear best (T) {e['no_best']}, no clear worst (B) {e['no_worst']}, all bad (G) "
                  f"{e['all_bad']}. Anchor rounds dropped: {s['anchor_rounds']}. Votes at half weight: "
                  f"{s['half_weight_votes']}. Marks left out (clip heard under the minimum in every attempt): "
                  f"{s['unheard_marks']}."]
    for name, _a, _b in POOLS:
        pooled = (doc.get("pooled") or {}).get(name)
        if pooled:
            L += ["", f"### {name} (pooled over both stages, the later stage's options)", ""] + _table(pooled)
    L += ["", "## Flags (section 3)", ""]
    if doc["flags"]:
        done = {n for n, b in doc["batches"].items() if b["status"] == "complete"}
        for f in doc["flags"]:
            if f["batch"] in done:
                what = (f"{f['field']} missing" if f["what"] == "missing"
                        else f"{f['bus']} spread {f['spread_lu']} LU > {f['limit_lu']}")
                L.append(f"- `{f['batch']}` {f['slot']} on {f['loop']} ({f['kind']}): {what}")
        hidden = sum(f["batch"] not in done for f in doc["flags"])
        if hidden:
            L.append(f"- {hidden} flag(s) in batches still being voted (named once they are done).")
    else:
        L.append("No round is flagged.")
    for s in doc["largest_spread"]:
        L.append(f"- Largest {s['bus']} spread in {s['stage']}: {s['spread_lu']} LU (limit {s['limit_lu']}).")
    moved = [m for m in doc["levels_moved"] if m["variant"]]
    if moved:
        L += [""] + [f"- Drift: `{m['variant']}` moved {m['moved_db']} dB ({m['file']})." for m in moved]
    L += ["", "## Hidden checks", ""]
    if doc["micro_sessions"]:
        L += ["| session | micro | anchors | caught | weight |", "|---|---|---|---|---|"]
        L += [f"| {g['session']} | {g['micro']} | {g['anchors']} | {g['caught']} | {g['weight']:g} |"
              for g in doc["micro_sessions"]]
    else:
        L.append("No anchor round has a vote yet.")
    r = doc["repeats"]
    L.append("")
    if r["voted"]:
        lo, hi = r["ci95"]
        L.append(f"Repeats (CD-47): {r['agree']} of {r['compared']} implied pairs agree "
                 f"(p {r['p']:.3f}, Wilson 95 % {lo:.3f}-{hi:.3f}) over {r['scored']} repeat(s); {r['missing']} first "
                 f"showing(s) not found. {r['caveat']}" if r["compared"] else
                 f"Repeats (CD-47): {r['voted']} voted, no implied pair to compare. {r['caveat']}")
        if r["too_soon"] or r["undated"]:         # D-013: at least min_days_after days after the first showing
            L.append(f"Left out of the pairs: {len(r['too_soon'])} repeat(s) voted under {r['min_days_after']} days "
                     f"after their first showing (D-013) and {len(r['undated'])} with a vote date that cannot be read.")
    else:                         # nothing here says which rounds come back or when (RATING.md: never told)
        L.append("Repeats (CD-47): none voted yet.")
    pb = doc["position_of_best"]["rows"]
    if pb:
        L += ["", "Position of best (descriptive), pooled over the shootout batches:", ""]
        for row in pb:
            p = f", chi-square {row['chi2']:.2f} on {row['df']} df, p {row['p']:.3f}" if row["chi2"] is not None else ""
            L.append(f"- {row['clips']} clips: tiles 1-{row['clips']} {row['counts']}{p}")
    L += ["", "## Rules and proposal", ""]
    L += _rules_md(doc)
    L += ["", "## Needs Shresth's word", ""]
    L += [f"- {'**Blocking:** ' if n['blocking'] else ''}{n['stage']}: {n['what']}" for n in doc["needs_word"]]
    L.append("- Every survivor: Shresth confirms them before the next job (section 1 step 1).")
    if doc["problems"]:
        L += ["", "## Problems", ""] + [f"- {p}" for p in doc["problems"]]
    return "\n".join(L) + "\n"


def _rules_md(doc: dict) -> list[str]:
    R, L = doc["rules"], []
    p2 = doc["proposal"]["day2"]
    if p2["status"] == "incomplete":
        wait = f" Waiting for every round of {', '.join(p2['waiting_for'])}." if p2.get("waiting_for") else ""
        L.append("After Day 1: not yet. " + ", ".join(f"{s} {st.replace('_', ' ')}" for s, st in p2["stages"].items())
                 + "." + wait)
    else:
        L.append("After Day 1 (job 2):")
        c = R.get("cut")
        d = (c or {}).get("decided")
        if c:
            add = f", {'+'.join(c['adds'])} included" if c.get("adds") else ""
            head = (f"- Cut list (CD-50): Day 1 took {_n(c['day1_minutes'])} min against the plan's "
                    f"{_n(c['day1_plan'])}, r = {c['r']:.3f}; Day 2's plan {_n(c['day2_plan'])} min (the rounds "
                    f"job 2 will draw{add})")
            if d is None:
                L.append(f"{head}, at r {_n(c['day2_minutes'])} min, Days 1-2 {_n(c['days12_minutes'])}; cuts: "
                         f"{', '.join(c['cuts']) or 'none'}.")
            else:                                         # his word decides; the list's reading stays as advice
                fit = "" if c["fits"] else "; it does not fit even so"
                L.append(f"{head}. Cuts: {', '.join(d['cuts']) or 'none'}, by {d['by']} (Day 2 at r "
                         f"{_n(d['day2_minutes'])} min, Days 1-2 {_n(d['days12_minutes'])}). The list read "
                         f"{', '.join(c['cuts']) or 'no cut'} (Day 2 at r {_n(c['day2_minutes'])} min, Days 1-2 "
                         f"{_n(c['days12_minutes'])}{fit}).")
        for key, label in (("K1", "Keys"), ("R1", "Kits"), ("R2", "Basses"), ("S1", "Sax")):
            if key not in R:                             # not run: the rule it needs raised
                L.append(f"- {label}: not applied (it needs the kits)")
                continue
            r = R[key]
            if "raised" in r:
                L.append(f"- {label}: **raised** {r['raised']}")
                continue
            if key == "K1":
                tie = f"; tie at third {r['tie_at_third']}" if r["tie_at_third"] else ""
                draw = f"; seeded draw {r['draw']['seed']} picked {r['draw']['picked']}" if r["draw"] else ""
                out = f"; cannot survive {r['out']}" if r["out"] else ""
                L.append(f"- Keys to K2 (CD-41): {', '.join(f'`{x}`' for x in r['k2'])}{tie}{draw}{out}; leader "
                         f"{_pick_text(r['leader'])}.")
            elif key == "R1":
                drop = f"; dropped `{r['dropped']}` (last, BW below 0)" if r["dropped"] else ""
                tie = "; a tie for last keeps all" if r["tie_for_last"] else ""
                out = f"; cannot survive {r['out']}" if r["out"] else ""
                L.append(f"- Kits (CD-42): {', '.join(f'`{x}`' for x in r['survivors'])}{drop}{tie}{out}; leader "
                         f"{_pick_text(r['leader'])}.")
            elif key == "R2":
                rl = r.get("relisten") or {}
                if rl.get("adds"):
                    on = " and ".join(f"`{k}` ({len(rl['rounds'][k])} round{'s' if len(rl['rounds'][k]) > 1 else ''})"
                                      for k in rl["kits"])
                    L.append(f"- Basses (CD-43): undecided ({'; '.join(rl['why'])}). Day 2 adds R2r, the bass "
                             f"re-listen (CD-54), on {on}; the band keeps `{rl['bass']}` meanwhile, and R2r decides "
                             "each kit's bass after Day 2.")
                else:
                    pairs = "; ".join(f"`{k}` takes `{p['bass']}` ({p['from']}{', ' + p['why'] if 'why' in p else ''})"
                                      for k, p in r["pairings"].items())
                    L.append(f"- Basses (CD-43): {pairs}. Each pairing rests on one round. Pooled R2 leader "
                             f"{_pick_text(r['pooled_leader'])}.")
            else:
                if not r.get("final", True):
                    L.append(f"- Sax: the final is cut, so S1 alone picks it: {_pick_text(r['leader'])}.")
                else:
                    three = " (a three-source final)" if r["three_source"] else ""
                    tie = f"; tie {r['tie']}" if r["tie"] else ""
                    draw = f"; seeded draw {r['draw']['seed']} picked {r['draw']['picked']}" if r["draw"] else ""
                    L.append(f"- Sax final (CD-18): {', '.join(f'`{x}`' for x in r['sax'])}{three}{tie}{draw}; leader "
                             f"{_pick_text(r['leader'])}.")
                for why in r["no_free_sax"]:
                    L.append(f"- **No free sax passes:** {why} (D-012's fallback is Shresth's call).")
        if p2["status"] == "ready":
            extra = ([f"`--add {'+'.join(p2['add'])}` (CD-54)"] if p2.get("add") else []) + (
                [f"`--cut {'+'.join(p2['cut'])}`" + (f" ({d['by']})" if d else "")] if p2["cut"] else [])
            cut = f" with {' and '.join(extra)}" if extra else ""
            prior = (f"; `--prior {'+'.join(p2['prior'])}`, the job that built the Day-1 batches (its drift keys, "
                     "section 3)" if p2.get("prior") else
                     "; **`--prior` missing:** no pulled summary built the Day-1 batches (see below)")
            L += ["", f"Job 2: `--leaders {p2['leaders']}`{cut}{prior}",
                  "", f"    {p2['submit']}", "",
                  "Checked: the string reads back through `draw.parse_leaders`, and `draw.day_band` builds the Day-2 "
                  f"band from it: {', '.join(f'{k} `{v}`' for k, v in p2['check']['day_band'].items() if v)}."]
        else:
            L.append(f"- Job 2's --leaders: not made ({p2['status']}).")
    later = [k for k in ("K2", "S1f", "S2a", "S2a" + RERUN, "S2b", "S2b" + RERUN, RELISTEN, "R3a", "R3b", "V", "W1",
                         "W2") if k in R]
    if later:
        L += ["", "Later days:"]
    for k in later:
        r = R[k]
        if "raised" in r:
            L.append(f"- {k}: **raised** {r['raised']}")
        elif k == "K2":
            L.append(f"- Keys survivors (CD-40, BW over K1 + K2): {', '.join(f'`{x}`' for x in r['survivors'])}; "
                     f"leader {_pick_text(r['leader'])}.")
        elif k == "S1f":
            L.append(f"- Sax final: winner `{r['winner']}` ({r['by']})" if r["winner"] else
                     f"- Sax final: tied {r['tied']} ({r['by']})")
        elif k.startswith("S2a"):
            ctrl = f"; `{CONTROL}` has the top BW" if r["control_top"] else ""
            L.append(f"- Warmth ({k}): {_pick_text(r['pick'])}{ctrl}.")
        elif k.startswith("S2b"):
            L.append(f"- Sax timing ({k}): {_pick_text(r['pick'])}.")
        elif k == RELISTEN:
            pairs = "; ".join(f"`{kit}` takes `{p['bass']}` ({p['from']}"
                              + (f", {p['why']}" if "why" in p else f", by {p['pick']['by']}") + ")"
                              for kit, p in r["pairings"].items())
            L.append(f"- Basses after R2r (CD-43, CD-54): {pairs}. Pooled R2r leader {_pick_text(r['pooled_leader'])}.")
        elif k == "R3a":
            L.append(f"- Swing (advisory, CD-13): stays `{r['pick']}`; BW leader {r['bw_leader']}; {r['note']}.")
        elif k == "R3b":
            L.append(f"- Humanize: {_pick_text(r['pick'])}.")
        elif k == "V":
            ranks = "; ".join(f"{m}: " + ", ".join(f"{p['place']}{'=' if p['tied'] else ''} `{p['option']}`"
                                                   for p in places) for m, places in r["moods"].items())
            c = r["borrowed"]
            L.append(f"- Vamps (seed data): {ranks}. Borrowed-chord condition (CD-45): "
                     f"{'met' if c['met'] else 'not met'} (" + ", ".join(
                         f"{x['vamp']} {x['needs']} in {m}: {'yes' if x['met'] else 'no'}"
                         for m, x in c["conditions"].items()) + ").")
        elif k == "W1":
            L.append(f"- Rain excerpt: {_pick_text(r['pick'])}.")
        elif k == "W2":
            L.append(f"- Rain level: {_pick_text(r['pick'])}" + ("; `rain_none` leads" if r["rain_none_leads"] else "")
                     + ".")
    p3 = (doc["proposal"].get("day3") or {})
    if p3.get("args"):              # the ARGS name the eligible stages, never a round (RATING.md: he is never told)
        c = R.get("day3_cut") or {}
        d = c.get("decided")
        cut = (f"; Day 3's plan {_n(c['day3_plan'])} min, at r {c['r']:.3f} {_n(c['day3_minutes'])} min, cuts "
               f"{', '.join((d or c)['cuts']) or 'none'}" + (f" ({d['by']})" if d else "") if c else "")
        L += ["", f"After Day 2 (job 3, {p3['status'].replace('_', ' ')}): `--leaders {p3['leaders']}`, "
                  f"`--prior {'+'.join(p3['prior'])}`" + (f", `--rerun-s2 {p3['rerun_s2']}`" if p3["rerun_s2"] else "")
              + cut + ".", "", f"    {p3['submit']}", "",
              "Checked: the string reads back through `draw.parse_leaders`, and `draw.day_band` builds the Day-3 band "
              f"from it: {', '.join(f'{k} `{v}`' for k, v in p3['check']['day_band'].items() if v)}."]
    elif p3.get("status") not in (None, "incomplete"):
        L += ["", f"After Day 2 (job 3): not made ({p3['status']})."]
    return L


# ------------------------------------------------------------------------------------------------ CLI
def _private(path: Path, inbox: Path, what: str) -> Path:
    """Tallies are as private as the votes: inside the repo only under data/ (gitignored), never in the inbox."""
    p = Path(path).expanduser().resolve()
    if _inside(p, inbox):
        raise TallyError(f"{what} {p} is inside the inbox {inbox}; tallies never sit next to clips")
    if _inside(p, REPO) and not _inside(p, REPO / "data"):
        raise TallyError(f"{what} {p} is inside the repo but not under data/ (gitignored); tallies are private until "
                         "Shresth confirms them (section 1 step 2)")
    return p


def _write_json(path: Path, doc: dict) -> None:
    atomic_write(path, (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                 .encode("utf-8"))


def write_outputs(doc: dict, batch_docs: Mapping[str, dict], out: Path, report: Path | None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    written = []
    for n, bd in batch_docs.items():
        _write_json(out / f"{n}.json", bd)
        written.append(out / f"{n}.json")
    _write_json(out / "shootouts.json", doc)
    written.append(out / "shootouts.json")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(report, report_md(doc, out_dir=shown_path(out)).encode("utf-8"))
        written.append(report)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.shootout.tally", description=__doc__.split("\n\n")[0])
    p.add_argument("--votes", required=True, type=Path, help="the votes folder (data/votes)")
    p.add_argument("--inbox", required=True, type=Path, help="the inbox root holding <date>/<batch>/")
    p.add_argument("--batch", nargs="+", action="extend", default=[], help="tally only these batches")
    p.add_argument("--out", type=Path, default=None, help="where the JSON goes (default <votes>/audit)")
    p.add_argument("--report", type=Path, default=None, help="write the markdown report here")
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--config", type=Path, default=None, help="configs/rating.yaml (heard.min_bars and the budget)")
    p.add_argument("--levels", type=Path, nargs="+", action="extend", default=[],
                   help="levels.json files whose drift moves go into the report (section 3)")
    p.add_argument("--runs", type=Path, default=REPO / "runs",
                   help="the pulled job summaries, runs/shootout-<job>/summary/ (job 2's --prior; default <repo>/runs)")
    p.add_argument("--day2-cuts", default=None, metavar="none|<cut>+<cut>",
                   help="Shresth's word on Day 2's cuts, which job 2's line then carries instead of the cut list's "
                        "(section 2.7): none, or Day-2 cut ids joined by +")
    p.add_argument("--day2-cuts-by", default=None, metavar="TEXT",
                   help="his words and their date, recorded with --day2-cuts (default: \"Shresth's word\")")
    p.add_argument("--serve-date", default=None, metavar="YYYY-MM-DD",
                   help="the day Day 3 will be served: job 3's --repeats name the stages voted at least 2 days before "
                        "it (section 2.5)")
    p.add_argument("--day3-cuts", default=None, metavar="none|<cut>+<cut>",
                   help="Shresth's word on Day 3's cuts (section 2.7): none, or Day-3 cut ids joined by +")
    p.add_argument("--day3-cuts-by", default=None, metavar="TEXT", help="his words and their date, with --day3-cuts")
    a = p.parse_args(argv)
    day2_cuts = None if a.day2_cuts is None else [] if a.day2_cuts == "none" else a.day2_cuts.split("+")
    day3_cuts = None if a.day3_cuts is None else [] if a.day3_cuts == "none" else a.day3_cuts.split("+")
    try:
        try:
            serve = None if a.serve_date is None else date.fromisoformat(a.serve_date)
        except ValueError:
            raise TallyError(f"--serve-date {a.serve_date!r} is not a date (YYYY-MM-DD)") from None
        if not a.votes.is_dir():
            raise TallyError(f"--votes {a.votes} is not a folder")
        if not a.inbox.is_dir():
            raise TallyError(f"--inbox {a.inbox} is not a folder")
        out = _private(a.out or a.votes / "audit", a.inbox, "--out")
        report = _private(a.report, a.inbox, "--report") if a.report else None
        cfg = load_config(a.config)
        plan = load_plan(a.plan)
        doc, batch_docs = tally(a.votes, a.inbox, plan=plan, batches=a.batch or None,
                                heard_min=cfg["heard"]["min_bars"], toss_min=cfg["heard"]["toss_min_bars"],
                                budget_min=cfg["daily_budget_minutes"], levels=a.levels, runs=a.runs,
                                day2_cuts=day2_cuts, day2_cuts_by=a.day2_cuts_by, serve_date=serve,
                                day3_cuts=day3_cuts, day3_cuts_by=a.day3_cuts_by)
        written = write_outputs(doc, batch_docs, out, report)
    except (TallyError, ConfigError, PlanError, DrawError, ManifestError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    for n, b in doc["batches"].items():
        print(f"{n}: {b['voted']} of {b['rounds']} rounds voted ({b['status'].replace('_', ' ')})")
    for sid, s in doc["stages"].items():
        print(f"stage {sid}: {s['status'].replace('_', ' ')} ({s['voted']} of {s['design_rounds']} design rounds)")
    p2 = doc["proposal"]["day2"]
    print(f"day 2 proposal: {p2['status']}" + (f": {p2['args']}" if p2["status"] == "ready" else ""))
    p3 = doc["proposal"].get("day3") or {}
    print(f"day 3 proposal: {p3.get('status')}" + (f": {p3['args']}" if p3.get("args") else ""))
    c = doc["rules"].get("cut") or {}
    if c.get("decided") is not None:
        print(f"day 2 cuts: {'+'.join(c['decided']['cuts']) or 'none'}, by {c['decided']['by']} (the cut list read "
              f"{'+'.join(c['cuts']) or 'none'})")
    for n in doc["needs_word"]:
        print(f"needs Shresth's word{' (blocking)' if n['blocking'] else ''}: {n['stage']}: {n['what']}")
    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
