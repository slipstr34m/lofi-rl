"""The shootout tally (docs/SHOOTOUTS.md section 7.9 acceptance tests, and section 8's rules).

Every batch here is built by the real draw and batch builder from fake renders (tests/helpers_shootout.py), so its
manifests are the job's own; every vote is synthetic, written by the rating page's own `Ledger` (src/lofi_rl/ui/
votes.py) on a fake clock into a temp votes folder, so each log line is in the page's event format. Nothing reads
data/votes or the real inbox."""
import json
import shlex
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from helpers_shootout import fake_renders
from lofi_rl.shootout import job
from lofi_rl.shootout import tally as T
from lofi_rl.shootout.batch import build
from lofi_rl.shootout.draw import day_band, day_minutes, draw, load_prior, parse_leaders, round_seconds, write_draw
from lofi_rl.shootout.plan import load_plan
from lofi_rl.ui.audit import agreement, heard_over_attempts, with_merged_heard
from lofi_rl.ui.manifest import HIDDEN_NAME, load_batch
from lofi_rl.ui.votes import Ledger, active_votes, read_all

ROOT = Path(__file__).resolve().parents[1]
PLAN = load_plan()
DAY1 = ("b1103-1-keys", "b1103-2-groove", "b1103-3-sax")
DAY2 = ("b1104-1-groove", "b1104-2-sax", "b1104-3-keys", "b1104-4-vamps")
DAY3 = ("b1106-1-mix", "b1106-2-mix")
DAY2_LEADERS = parse_leaders("keys=gu_tine_ep+fp_fm_ep1+gu_organ/kit=kit_swirly/bass=bass_fp_finger_yr/"
                             "sax=sax_mtg_tenor+sax_gu_alto")
DAY3_LEADERS = parse_leaders("keys=fp_fm_ep1/kit=kit_swirly/bass=bass_fp_finger_yr/sax=sax_mtg_tenor")
ALL_STAGES = ["K1", "R1", "R2", "S1", "R3a", "R3b", "S1f", "S2a", "S2b", "K2", "V"]
# One taste for every stage, most liked first: in each round the best is the most liked clip and the worst the least.
ORDER = ["fp_fm_ep1", "gs_wurli_ep200", "jrhodes3d", "gu_tine_ep", "fp_upright_kw", "vcsl_upright_knight",
         "fp_fsbs_jazz_guitar", "gu_organ",
         "kit_swirly", "kit_virtuosity", "kit_gu_jazz",
         "bass_fp_finger_yr", "bass_vsco2_pizz", "bass_gu_acoustic",
         "sax_mtg_tenor", "sax_gu_alto", "sax_fp_tenor", "sax_weresax_alto", "sax_mtg_alto", "sax_gu_tenor"]
FULL = "11111111"
PST = timezone(timedelta(hours=-8))


class Clock:
    def __init__(self, t: datetime = datetime(2026, 11, 3, 20, 0, tzinfo=PST)):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def _day(tmp: Path, day: str, seed: int, **kw) -> dict[str, Path]:
    run = tmp / "runs" / f"shootout-{seed}"
    doc = draw(PLAN, day, seed=seed, **kw)
    write_draw(doc, run / "summary" / "draw.json")
    fake_renders(doc, run / "renders")
    folders = build(doc, run / "renders", tmp / "inbox", summary_dir=run / "summary", plan=PLAN, check=False)
    return {f.name: f for f in folders}


@pytest.fixture(scope="module")
def inbox(tmp_path_factory) -> dict:
    """All three days' batches, built once (read-only for the tests: a test that edits one copies it)."""
    tmp = tmp_path_factory.mktemp("tally")
    folders = _day(tmp, "day1", 0x596A)
    folders |= _day(tmp, "day2", 0x596B, leaders=DAY2_LEADERS)
    priors = [load_prior(tmp / "runs" / f"shootout-{s}" / "summary" / "batches.json") for s in (0x596A, 0x596B)]
    folders |= _day(tmp, "day3", 0x596C, leaders=DAY3_LEADERS, priors=priors, repeats=ALL_STAGES)
    return {"root": tmp / "inbox", "folders": folders}


def prefer(order=ORDER, *, catch: bool = True):
    """A chooser: best the most liked clip, worst the least liked; an anchor's sour copy is worst when `catch`."""
    rank = {o: i for i, o in enumerate(order)}

    def choose(r, hr):
        opt = {c: hr.clips[c]["option"] for c in r.ids}
        ids = sorted(r.ids, key=lambda c: (rank.get(opt[c], len(rank)), c))
        if hr.kind == "anchor":
            rest = [c for c in ids if c != hr.anchor_clip]
            return {"best": rest[0], "worst": hr.anchor_clip if catch else rest[-1]}
        return {"best": ids[0], "worst": ids[-1]}
    return choose


def vote(folder: Path, votes: Path, clock: Clock, choose, *, rounds=None, commit_ms: int = 60000,
         redo: dict | None = None) -> None:
    """Vote rounds of one batch through the page's Ledger. `choose(round, hidden round)` gives the picks, and may add
    `heard` ({opaque id: heard_bars}) and `gap` (time since the last event). `redo` ({round id: "undo" | "abandon"})
    first answers that round and takes the vote back (Backspace twice, why "undo") or leaves it (Esc, an `abandon`
    with what was heard), with `heard`'s bars, and then answers it again with every clip at "00000000", as the page
    shows a round whose bar count restarted (D-013 as amended: the bars of every attempt count)."""
    m, h = load_batch(folder, check_files=False)
    led = Ledger(votes, m, h, clock=clock)
    try:
        for r in m.rounds if rounds is None else [m.round(x) for x in rounds]:
            p = dict(choose(r, h.rounds[r.id]))
            clock.t += p.pop("gap", timedelta(seconds=90))
            heard = p.pop("heard", {})
            clips = [{"clip": c, "pos": i + 1, "heard_bars": heard.get(c, FULL), "heard_ms": 20000}
                     for i, c in enumerate(r.ids)]
            how = (redo or {}).get(r.id)
            if how == "undo":
                first = led.submit({"type": "shootout", "round": r.id, **p, "t_ms": {"commit": commit_ms},
                                    "clips": clips})
                led.submit({"type": "retract", "target": first["event_id"], "why": "undo"})
            elif how == "abandon":
                led.submit({"type": "abandon", "round": r.id, "why": "esc", "clips": clips})
            if how:
                clips = [dict(c, heard_bars="00000000") for c in clips]
            led.submit({"type": "shootout", "round": r.id, **p, "t_ms": {"commit": commit_ms}, "clips": clips})
    finally:
        led.close()


def vote_days(inbox: dict, votes: Path, names=DAY1, choose=None, **kw) -> Clock:
    votes.mkdir(exist_ok=True)
    clock = Clock()
    for n in names:
        vote(inbox["folders"][n], votes, clock, choose or prefer(), **kw)
    return clock


def tally(inbox: dict, votes: Path, **kw):
    kw.setdefault("runs", inbox["root"].parent / "runs")         # the fixture's job summaries, shootout-<seed>
    return T.tally(votes, inbox["root"], plan=PLAN, today=date(2026, 11, 3), **kw)


JOB1 = str(0x596A)                 # the fixture's Day-1 run is runs/shootout-<seed>, as the cluster's is shootout-<job>


def _row(option, best=0, worst=0, heard=4, all_bad=0, rounds=None) -> T.Row:
    r = T.Row(option, shown=heard, heard_n=heard, heard=Fraction(heard), best=Fraction(best), worst=Fraction(worst),
              rounds=Fraction(heard if rounds is None else rounds), all_bad=Fraction(all_bad), all_bad_n=all_bad)
    return r


def _rows(*rows: T.Row) -> dict[str, T.Row]:
    return {r.option: r for r in rows}


# ---------------------------------------------------------------------------------------------- section 7.9 tests
def test_counts_equal_the_audits_anchor_rounds_excepted(inbox, tmp_path):
    votes = tmp_path / "votes"
    vote_days(inbox, votes)
    doc, batches = tally(inbox, votes)
    for name in DAY1:
        audit = {o["option"]: (o["shown"], o["best"], o["worst"])
                 for o in batches[name]["audit"]["shootout"]["options"]}
        got: dict[str, list[int]] = {}
        for sid in {r["stage"] for r in batches[name]["votes"]}:
            for o in doc["stages"][sid]["options"]:
                got[o["option"]] = [o["shown"], int(o["best"]), int(o["worst"])]
                assert o["net"] == o["audit"]["best"] - o["audit"]["worst"]
        anchors = [r for r in batches[name]["votes"] if r["kind"] == "anchor"]
        for r in anchors:                               # the audit counts an anchor round, its broken clip skipped
            for c in r["clips"]:
                if c["option"] == r["anchor"]:          # a round's options are distinct (section 2.5)
                    continue
                got[c["option"]][0] += 1
                got[c["option"]][1] += int(c["option"] == r["best"])
                got[c["option"]][2] += int(c["option"] == r["worst"])
        assert {k: tuple(v) for k, v in got.items()} == audit, name
        assert len(anchors) == (1 if name == "b1103-3-sax" else 0)
    assert doc["stages"]["S1"]["anchor_rounds"] == 1 and doc["stages"]["S1"]["design_rounds"] == 6
    assert sum(o["shown"] for o in doc["stages"]["K1"]["options"]) == 24      # 6 rounds of 4, every clip heard


def test_at_an_explicit_4_bar_minimum_a_mark_on_a_clip_heard_3_bars_is_left_out(inbox, tmp_path):
    """The heard filter's mechanism at the old 4-bar minimum (CD-38 before D-013's amendment): the recount leaves out
    a best on a clip heard 3 bars, the audit counts it. Under the rule as amended (1 bar, the default) it counts."""
    votes = tmp_path / "votes"
    votes.mkdir()
    base = prefer()
    first = {}

    def choose(r, hr):
        p = base(r, hr)
        if not first:                                    # the first round: its best was heard 3 bars only
            first.update(round=r.id, option=hr.clips[p["best"]]["option"])
            p["heard"] = {p["best"]: "11100000"}
        return p

    vote(inbox["folders"]["b1103-1-keys"], votes, Clock(), choose)
    now, _ = tally(inbox, votes, batches=["b1103-1-keys"])
    assert now["heard_min_bars"] == 1 and now["stages"]["K1"]["unheard_marks"] == 0
    doc, batches = tally(inbox, votes, batches=["b1103-1-keys"], heard_min=4, toss_min=2)
    row = next(o for o in doc["stages"]["K1"]["options"] if o["option"] == first["option"])
    audit = next(o for o in batches["b1103-1-keys"]["audit"]["shootout"]["options"] if o["option"] == first["option"])
    assert audit["best"] == row["audit"]["best"] == int(row["best"]) + 1            # the audit counts it; BW does not
    assert row["heard_n"] == row["shown"] - 1 and row["unheard_marks"] == 1
    assert row["bw"] == pytest.approx((row["best"] - row["worst"]) / row["heard"])
    assert doc["stages"]["K1"]["unheard_marks"] == 1
    assert batches["b1103-1-keys"]["audit"]["heard"]["heard"] == 23                  # the audit's own heard count


def test_a_retracted_vote_stops_counting_and_its_new_answer_counts(inbox, tmp_path):
    votes = tmp_path / "votes"
    votes.mkdir()
    folder = inbox["folders"]["b1103-1-keys"]
    vote(folder, votes, Clock(), prefer())
    before, _ = tally(inbox, votes, batches=["b1103-1-keys"])
    m, h = load_batch(folder, check_files=False)
    r = m.rounds[0]
    p = prefer()(r, h.rounds[r.id])
    led = Ledger(votes, m, h, clock=Clock(datetime(2026, 11, 3, 21, 0, tzinfo=PST)))
    try:                                                  # Backspace twice on the page: the vote is taken back
        led.submit({"type": "retract", "target": active_votes(led.snapshot())[r.id]["event_id"]})
        led.submit({"type": "shootout", "round": r.id, "best": p["worst"], "worst": p["best"],
                    "clips": [{"clip": c, "pos": i + 1, "heard_bars": FULL} for i, c in enumerate(r.ids)]})
    finally:
        led.close()
    after, batches = tally(inbox, votes, batches=["b1103-1-keys"])
    old = {o["option"]: o for o in before["stages"]["K1"]["options"]}
    new = {o["option"]: o for o in after["stages"]["K1"]["options"]}
    hi, lo = (h.rounds[r.id].clips[p[k]]["option"] for k in ("best", "worst"))
    assert (new[hi]["best"], new[hi]["worst"]) == (old[hi]["best"] - 1, old[hi]["worst"] + 1)
    assert (new[lo]["best"], new[lo]["worst"]) == (old[lo]["best"] + 1, old[lo]["worst"] - 1)
    assert batches["b1103-1-keys"]["status"] == "complete" and batches["b1103-1-keys"]["audit"]["retracted"] == 1


def test_repeats_are_scored_for_agreement_and_left_out_of_the_counts(inbox, tmp_path):
    votes = tmp_path / "votes"
    clock = vote_days(inbox, votes, DAY1 + DAY2)
    flipped = []

    def choose(r, hr):                                   # the same taste, except one repeat answered the other way
        p = prefer()(r, hr)
        if hr.kind == "repeat" and not flipped:
            flipped.append(r.id)
            p = {"best": p["worst"], "worst": p["best"]}
        return p

    for n in DAY3:
        vote(inbox["folders"][n], votes, clock, choose)
    doc, _ = tally(inbox, votes)
    before, _ = tally(inbox, votes, batches=list(DAY1 + DAY2))
    for sid in ALL_STAGES:                               # the repeats change no stage's counts
        assert doc["stages"][sid]["options"] == before["stages"][sid]["options"], sid
    rep = doc["repeats"]
    assert rep["voted"] == rep["found"] == 5 and rep["missing"] == 0 and rep["pending"] == 0
    logs = read_all(votes)
    originals = {k: with_merged_heard(v, heard_over_attempts(evs)) for evs in logs.values()
                 for k, v in active_votes(evs).items()}
    agree = compared = 0
    for row in rep["rows"]:                              # the audit's own figure, pooled
        a, n = agreement("shootout", originals[row["of"]], originals[row["round"]], 1)
        assert (row["agree"], row["compared"]) == (a, n)
        agree, compared = agree + a, compared + n
    assert (rep["agree"], rep["compared"]) == (agree, compared) and 0 < agree < compared
    lo, hi = rep["ci95"]
    assert lo < rep["p"] < hi and rep["p"] == agree / compared
    assert {r["stage"] for r in rep["rows"]} & {"K1", "K2"}                          # one per group (section 2.5)
    assert doc["stages"]["K1"]["repeat_rounds"] + doc["stages"]["K2"]["repeat_rounds"] == 1
    assert "What could give a repeat away" in rep["caveat"]


def test_a_missed_anchor_halves_its_micro_sessions_weight(inbox, tmp_path):
    folder = inbox["folders"]["b1103-3-sax"]
    m, h = load_batch(folder, check_files=False)
    anchor = next(r.id for r in m.rounds if h.rounds[r.id].kind == "anchor")
    first = next(r.id for r in m.rounds if h.rounds[r.id].kind == "none")
    rest = [r.id for r in m.rounds if r.id not in (anchor, first)]
    votes = tmp_path / "votes"
    votes.mkdir()
    clock = Clock()
    vote(folder, votes, clock, prefer(catch=False), rounds=[first, anchor])       # micro 1 misses its anchor
    clock.t += timedelta(minutes=11)                                              # a new micro-session (RATING.md)
    vote(folder, votes, clock, prefer(), rounds=rest)
    doc, batches = tally(inbox, votes, batches=["b1103-3-sax"])
    groups = doc["micro_sessions"]
    assert [(g["micro"], g["anchors"], g["caught"], g["weight"]) for g in groups] == [(1, 1, 0, 0.5)]
    w = {r["round"]: r["weight"] for r in batches["b1103-3-sax"]["votes"]}
    assert w[first] == w[anchor] == 0.5 and all(w[x] == 1.0 for x in rest)
    assert doc["stages"]["S1"]["half_weight_votes"] == 1
    best = next(r for r in batches["b1103-3-sax"]["votes"] if r["round"] == first)["best"]
    row = next(o for o in doc["stages"]["S1"]["options"] if o["option"] == best)
    full = sum(1 for r in batches["b1103-3-sax"]["votes"] if r["kind"] == "none" and r["best"] == best) - 1
    assert row["best"] == full + 0.5                                                # its best counts half
    assert row["heard"] == row["shown"] - 0.5                                       # one appearance at half weight


def test_the_day2_proposal_waits_for_every_round_of_day1_since_a_late_anchor_reweighs(inbox, tmp_path):
    votes = tmp_path / "votes"
    clock = vote_days(inbox, votes, names=DAY1[:2])
    folder = inbox["folders"]["b1103-3-sax"]
    m, h = load_batch(folder, check_files=False)
    anchor = next(r.id for r in m.rounds if h.rounds[r.id].kind == "anchor")
    vote(folder, votes, clock, prefer(), rounds=[r.id for r in m.rounds if r.id != anchor])
    doc, _ = tally(inbox, votes)
    assert doc["stages"]["S1"]["status"] == "complete" and not doc["stages"]["S1"]["ready"]
    assert doc["proposal"]["day2"]["status"] == "incomplete"
    assert doc["proposal"]["day2"]["waiting_for"] == ["b1103-3-sax"]
    assert "its rules wait for the rest of b1103-3-sax" in T.report_md(doc)
    assert doc["stages"]["R1"]["half_weight_votes"] == 0
    vote(folder, votes, clock, prefer(catch=False), rounds=[anchor])       # missed, in the same micro-session
    doc, _ = tally(inbox, votes)
    assert doc["proposal"]["day2"]["status"] == "ready"
    assert [(g["anchors"], g["caught"], g["weight"]) for g in doc["micro_sessions"]] == [(1, 0, 0.5)]
    assert doc["stages"]["R1"]["half_weight_votes"] == 3 and doc["stages"]["K1"]["half_weight_votes"] == 6


def test_the_leaders_string_round_trips_through_parse_leaders_and_the_day2_band(inbox, tmp_path):
    votes = tmp_path / "votes"
    vote_days(inbox, votes)
    doc, _ = tally(inbox, votes)
    p = doc["proposal"]["day2"]
    assert p["status"] == "ready" and p["cut"] == [] and not p["blocking"]       # 19 min of 23 planned: r 0.826
    assert p["prior"] == [JOB1] and p["args"] == f"--stage day2 --leaders {p['leaders']} --prior {JOB1}"
    text = p["leaders"]
    leaders = parse_leaders(text)
    assert T.leaders_string(leaders) == text and list(leaders) == ["keys", "kit", "bass", "sax"]
    assert leaders["keys"][0] == "fp_fm_ep1" and len(leaders["keys"]) == 3        # liked best in each of its rounds
    assert leaders["kit"] == ("kit_swirly",) and leaders["bass"] == ("bass_fp_finger_yr",)
    assert leaders["sax"][0] == "sax_mtg_tenor" and len(leaders["sax"]) == 2
    k1 = {o["option"]: o["bw"] for o in doc["stages"]["K1"]["options"]}
    assert sorted(k1, key=lambda o: -k1[o])[:3] == sorted(leaders["keys"], key=lambda o: -k1[o])
    assert doc["rules"]["R1"]["dropped"] == "kit_gu_jazz"                       # last with BW -1 (CD-42)
    assert doc["rules"]["R1"]["survivors"] == ["kit_swirly", "kit_virtuosity"]
    assert {v["bass"] for v in doc["rules"]["R2"]["pairings"].values()} == {"bass_fp_finger_yr"}
    band = day_band(PLAN, "day2", leaders, [])
    assert band["keys"] == "fp_fm_ep1" and band["sax"] == "sax_mtg_tenor" and band["kit"] == "kit_swirly"
    assert p["check"]["day_band"] == band
    draw(PLAN, "day2", seed=7, leaders=leaders)                                    # the draw itself takes it
    run = tmp_path / "runs" / "shootout-13640002"
    assert job.main(["--run", str(run), "--inbox", str(tmp_path / "inbox2"), *shlex.split(p["args"]),
                     "--dry-run"]) == 0                                            # and so does the job's own check
    written = json.loads((run / "summary" / "leaders.json").read_text())
    assert written["leaders"] == {k: list(v) for k, v in leaders.items()}
    assert written["prior"] == [JOB1] and written["prior_runs"] == [str(run.parent / f"shootout-{JOB1}")]


def test_job_2s_args_name_the_job_that_built_the_day1_batches_as_prior(inbox, tmp_path):
    """Section 3 "Drift between jobs": job 2 compares its drift keys with job 1's, so its ARGS carry --prior <job 1>
    (jobs/shootout.sbatch's Day-2 form). The tally finds job 1 in the pulled summaries: the batches.json whose `run`
    is shootout-<job> and which lists every Day-1 batch with its manifest's round ids."""
    votes = tmp_path / "votes"
    vote_days(inbox, votes)
    runs = inbox["root"].parent / "runs"
    found = T.built_by(runs, {n: [r.id for r in load_batch(inbox["folders"][n], check_files=False)[0].rounds]
                              for n in DAY1 + DAY2})
    assert {n: found[n] for n in DAY1} == dict.fromkeys(DAY1, [JOB1])
    assert {n: found[n] for n in DAY2} == dict.fromkeys(DAY2, [str(0x596B)])
    doc, _ = tally(inbox, votes)
    assert doc["batches"]["b1103-1-keys"]["jobs"] == [JOB1] and doc["runs"] == str(runs)
    md = T.report_md(doc)
    assert f"--prior {JOB1}'" in md and f"`--prior {JOB1}`, the job that built the Day-1 batches" in md
    # no summary pulled (or none that built these batches): the leaders stand, the --prior is put to Claude/Shresth
    other = tmp_path / "runs"
    doc0, _ = tally(inbox, votes, runs=other)
    p0 = doc0["proposal"]["day2"]
    assert p0["status"] == "ready" and p0["prior"] == [] and "--prior" not in p0["args"]
    assert p0["leaders"] == doc["proposal"]["day2"]["leaders"]
    assert any(n["stage"] == "prior" and n["blocking"] for n in doc0["needs_word"]) and p0["blocking"]
    assert "**`--prior` missing:**" in T.report_md(doc0)
    # a dry run's summary of the same batch names, with its own random round ids, is not job 1; nor is a run that
    # is not shootout-<job id>
    src = json.loads((runs / f"shootout-{JOB1}" / "summary" / "batches.json").read_text())
    fake = json.loads(json.dumps(src))
    for b in fake["batches"].values():
        for r in b["rounds"]:
            r["id"] = "r" + "0" * 8
    for name, doc_ in ((f"shootout-{JOB1}", fake), ("shootout-day1", dict(src, run="shootout-day1"))):
        (other / name / "summary").mkdir(parents=True)
        (other / name / "summary" / "batches.json").write_text(json.dumps(doc_))
    assert T.tally(votes, inbox["root"], plan=PLAN, runs=other)[0]["proposal"]["day2"]["prior"] == []
    (other / "shootout-13650001" / "summary").mkdir(parents=True)                 # job 1's summary, pulled
    (other / "shootout-13650001" / "summary" / "batches.json").write_text(json.dumps(dict(src, run="shootout-13650001")))
    p1 = T.tally(votes, inbox["root"], plan=PLAN, runs=other)[0]["proposal"]["day2"]
    assert p1["prior"] == ["13650001"] and p1["args"].endswith("--prior 13650001") and not p1["blocking"]


def test_the_cut_list_follows_the_day1_minutes_and_the_final_cut_leaves_one_sax(inbox, tmp_path):
    assert T.day2_cut(PLAN, 19.0)["cuts"] == []
    assert T.day2_cut(PLAN, 25.3)["cuts"] == ["vamps-high"]                       # r 1.1: 31.0 min on Day 2
    assert T.day2_cut(PLAN, 27.6)["cuts"] == ["vamps-high", "latin3"]             # r 1.2: Days 1-2 57.1, then 53.5
    c = T.day2_cut(PLAN, 46.0)                                                    # r 2: nothing fits
    assert c["cuts"] == ["vamps-high", "latin3", "final"] and not c["fits"] and c["day2_minutes"] == 39.2
    assert c["day2_plan"] == 28.2 and c["saves_min"] == 8.6        # the drawn plan's rounds, each anchor at its mean
    votes = tmp_path / "votes"
    vote_days(inbox, votes, commit_ms=92000)       # 9.2 + 9.2 + 10.7 = 29.1 min: r 1.265
    doc, _ = tally(inbox, votes)
    p = doc["proposal"]["day2"]
    assert doc["rules"]["cut"]["day1_minutes"] == 29.1
    assert p["cut"] == ["vamps-high", "latin3", "final"] and p["args"].endswith("--cut vamps-high+latin3+final")
    leaders = parse_leaders(p["leaders"])
    assert leaders["sax"] == ("sax_mtg_tenor",)                                    # S1 alone picks the sax
    assert day_band(PLAN, "day2", leaders, p["cut"])["sax"] == "sax_mtg_tenor"
    with pytest.raises(Exception, match="takes"):                                  # two ids with the final cut
        day_band(PLAN, "day2", {**leaders, "sax": ("sax_mtg_tenor", "sax_gu_alto")}, p["cut"])


def test_ties_at_third_place_take_up_to_4_and_then_a_seeded_draw():
    a, b = _row("a", 3, 0), _row("b", 2, 0)
    two = T.keys_to_k2(_rows(a, b, _row("c", 1, 1), _row("d", 1, 1), _row("e", 0, 3)), 1, default="gu_tine_ep")
    assert two["k2"] == ["a", "b", "c", "d"] and two["tie_at_third"] == ["c", "d"] and two["draw"] is None
    three = _rows(a, b, _row("c", 1, 1), _row("d", 1, 1), _row("e", 1, 1), _row("f", 0, 3))
    got = T.keys_to_k2(three, 1, default=None)
    assert len(got["k2"]) == 4 and got["k2"][:2] == ["a", "b"] and set(got["k2"][2:]) < {"c", "d", "e"}
    assert got["draw"]["among"] == ["c", "d", "e"] and got["draw"]["picked"] == got["k2"][2:]
    assert T.keys_to_k2(three, 1, default=None)["k2"] == got["k2"]                 # one seed, one draw
    assert len({tuple(T.keys_to_k2(three, s, default=None)["k2"]) for s in range(12)}) > 1
    exact = T.keys_to_k2(_rows(a, b, _row("c", 2, 1), _row("d", 1, 1)), 1, default=None)
    assert exact["k2"] == ["a", "b", "c"] and exact["tie_at_third"] is None       # c beats d on bests (CD-39)
    lead = T.keys_to_k2(_rows(_row("a", 3, 0), _row("b", 3, 0), _row("c", 1, 1), _row("gu_tine_ep", 3, 0)), 1,
                        default="gu_tine_ep")
    assert lead["leader"]["pick"] == "gu_tine_ep" and lead["k2"][0] == "gu_tine_ep"   # the default among the tied
    nodef = T.keys_to_k2(_rows(_row("a", 3, 0), _row("b", 3, 0), _row("c", 1, 1)), 1, default="gu_tine_ep")
    assert nodef["leader"]["needs_word"] and nodef["leader"]["draw"]["among"] == ["a", "b"]
    # the sax: two tied for the second place go to a seeded draw; three tied give a three-source final
    s1 = [_row("x", 4, 0), _row("y", 1, 1), _row("z", 1, 1), _row("w", 0, 2)]
    fin = T.sax_to_final(_rows(*s1), [], 3, default=None)
    assert fin["sax"][0] == "x" and len(fin["sax"]) == 2 and fin["draw"]["among"] == ["y", "z"]
    tri = T.sax_to_final(_rows(_row("x", 2, 0), _row("y", 2, 0), _row("z", 2, 0), _row("w", 0, 2)), [], 3,
                         default="z")
    assert tri["three_source"] and sorted(tri["sax"]) == ["x", "y", "z"] and tri["sax"][0] == "z"
    assert day_band(PLAN, "day2", {**DAY2_LEADERS, "sax": ("sax_gu_tenor", "sax_mtg_tenor", "sax_gu_alto")})


def test_an_option_all_bad_in_half_its_rounds_cannot_survive(inbox, tmp_path):
    rows = _rows(_row("a", 3, 0, all_bad=0), _row("b", 1, 0, heard=3, all_bad=2, rounds=3), _row("c", 1, 1),
                 _row("d", 0, 1), _row("e", 0, 2))
    assert rows["b"].out and not rows["a"].out and rows["b"].bw > rows["c"].bw
    k2 = T.keys_to_k2(rows, 1, default=None)
    assert "b" not in k2["k2"] and k2["out"] == ["b"] and k2["k2"] == ["a", "c", "d"]
    assert not _row("q", heard=3, all_bad=1, rounds=3).out and _row("q", heard=4, all_bad=2, rounds=4).out
    # through the votes: G on two K1 rounds that share a family's twins (they meet in all three of their rounds)
    folder = inbox["folders"]["b1103-1-keys"]
    m, h = load_batch(folder, check_files=False)
    opts = {r.id: {h.rounds[r.id].clips[c]["option"] for c in r.ids} for r in m.rounds}
    twin = next(o for o in ORDER[:8] if sum(o in s for s in opts.values()) == 3)
    bad = [rid for rid, s in opts.items() if twin in s][:2]
    base = prefer()

    def choose(r, hr):
        return {"all_bad": True, "worst": base(r, hr)["worst"]} if r.id in bad else base(r, hr)

    votes = tmp_path / "votes"
    vote_days(inbox, votes, choose=choose)
    doc, _ = tally(inbox, votes)
    k1 = {o["option"]: o for o in doc["stages"]["K1"]["options"]}
    out = sorted(o for o, row in k1.items() if row["cannot_survive"])
    assert twin in out and len(out) >= 2 and all(k1[o]["all_bad"] >= 2 for o in out)
    assert doc["stages"]["K1"]["escape"]["all_bad"] == 2
    assert not set(out) & set(doc["rules"]["K1"]["k2"]) and doc["rules"]["K1"]["out"] == out
    md = T.report_md(doc)
    assert md.count("(cannot survive, CD-44)") >= len(out)
    # every kit is in every R1 round: G on two of the three leaves no kit, which the spec does not decide
    gm, gh = load_batch(inbox["folders"]["b1103-2-groove"], check_files=False)
    r1 = [r.id for r in gm.rounds if gh.rounds[r.id].clips[r.ids[0]]["stage"] == "R1"][:2]
    votes2 = tmp_path / "votes2"
    vote_days(inbox, votes2, choose=lambda r, hr: {"all_bad": True} if r.id in r1 else base(r, hr))
    doc2, _ = tally(inbox, votes2)
    assert "raised" in doc2["rules"]["R1"] and "section 8 Kits" in doc2["rules"]["R1"]["raised"]
    assert doc2["proposal"]["day2"]["status"] == "blocked"
    assert any(n["blocking"] and n["stage"] == "R1" for n in doc2["needs_word"])
    md2 = T.report_md(doc2)
    assert "- Kits: **raised** SHOOTOUTS.md section 8 Kits" in md2 and "Basses: not applied" in md2
    assert "Job 2's --leaders: not made (blocked)" in md2


def test_a_batch_with_no_votes_yet_is_reported_as_such(inbox, tmp_path):
    votes = tmp_path / "votes"
    vote_days(inbox, votes, names=("b1103-1-keys",))
    doc, batches = tally(inbox, votes, batches=list(DAY1))
    assert batches["b1103-1-keys"]["status"] == "complete"
    for n in ("b1103-2-groove", "b1103-3-sax"):
        assert batches[n]["status"] == "no_votes" and batches[n]["voted"] == 0 and batches[n]["audit"] is None
        assert doc["batches"][n]["minutes"] is None
    for sid in ("R1", "R2", "S1"):
        assert doc["stages"][sid]["status"] == "no_votes" and doc["stages"][sid]["options"] is None
    assert doc["proposal"]["day2"] == {"status": "incomplete", "stages": {
        "K1": "complete", "R1": "no_votes", "R2": "no_votes", "S1": "no_votes"},
        "waiting_for": ["b1103-2-groove", "b1103-3-sax"]}
    md = T.report_md(doc)
    assert md.count("No votes yet") == 3 and "| `b1103-2-groove` | 2026-11-03 | 6 | 0 | no votes yet | - |" in md
    assert "After Day 1: not yet." in md
    k1 = md.split("### K1")[1].split("###")[0]
    assert k1.count("| `") == 8 and "No votes yet" not in k1
    part = tmp_path / "part"
    part.mkdir()
    groove = inbox["folders"]["b1103-2-groove"]
    vote(groove, part, Clock(), prefer(), rounds=[r.id for r in load_batch(groove, check_files=False)[0].rounds][:2])
    doc3, _ = tally(inbox, part, batches=["b1103-2-groove"])
    assert doc3["batches"]["b1103-2-groove"]["status"] == "partial"
    assert "Partial:" in T.report_md(doc3)


# ---------------------------------------------------------------------------------------------- the later rules
def test_keys_survivors_take_the_top_2_and_a_third_above_0():
    pooled = _rows(_row("a", 4, 0), _row("b", 2, 1), _row("c", 1, 0), _row("x", 6, 0))
    assert T.keys_survivors(pooled, ["a", "b", "c"], 1, default=None)["survivors"] == ["a", "b", "c"]
    neg = _rows(_row("a", 4, 0), _row("b", 2, 1), _row("c", 0, 1))
    assert T.keys_survivors(neg, ["a", "b", "c"], 1, default=None)["survivors"] == ["a", "b"]
    both = _rows(_row("a", 4, 0), _row("b", 2, 1), _row("c", 2, 1))                # b and c tie with BW above 0
    assert sorted(T.keys_survivors(both, ["a", "b", "c"], 1, default=None)["survivors"]) == ["a", "b", "c"]
    with pytest.raises(T.RuleGap, match="section 8 Keys .CD-40."):
        T.keys_survivors(_rows(_row("a", 4, 0), _row("b", 1, 2), _row("c", 1, 2)), ["a", "b", "c"], 1, default=None)
    with pytest.raises(T.RuleGap, match="third survivor place"):
        T.keys_survivors(_rows(_row("a", 4, 0), _row("b", 3, 0), _row("c", 2, 1), _row("d", 2, 1)),
                         ["a", "b", "c", "d"], 1, default=None)


def test_the_sax_final_goes_to_bests_then_combined_bw_then_shresth():
    pooled = _rows(_row("f1", 3, 1), _row("f2", 4, 0))
    assert T.sax_final(_rows(_row("f1", 2, 1), _row("f2", 1, 2)), pooled, ["f1", "f2"], s1_leader="f1")[
        "winner"] == "f1"
    tied = T.sax_final(_rows(_row("f1", 2, 1), _row("f2", 2, 1)), pooled, ["f1", "f2"], s1_leader="f1")
    assert tied["winner"] == "f2" and tied["rerun_s2"] == "f2" and "combined" in tied["by"]
    even = T.sax_final(_rows(_row("f1", 2, 1), _row("f2", 2, 1)), _rows(_row("f1", 2, 0), _row("f2", 2, 0)),
                       ["f1", "f2"], s1_leader="f1")
    assert even["winner"] is None and even["needs_word"] and even["tied"] == ["f1", "f2"]


def test_the_variant_vamp_and_rain_rules():
    w = T.warmth(_rows(_row("sax_plain", 1, 1), _row("sax_warm_on", 1, 1), _row("sax_expr_off", 3, 0)), 1,
                 default="sax_plain")
    assert w["pick"]["pick"] == "sax_plain" and w["control_top"]                    # the default among the tied
    t = T.timing(_rows(_row("timing_grid", 0, 2), _row("timing_lag", 2, 0), _row("timing_swung", 2, 0)), 1,
                 default="timing_grid")
    assert t["pick"]["pick"] == "timing_grid" and t["pick"]["needs_word"]          # a tie without the default
    s = T.swing(_rows(_row("swing_50", 3, 0), _row("swing_55", 0, 1)), default="swing_55")
    assert s["pick"] == "swing_55" and s["bw_leader"] == ["swing_50"]
    assert T.humanize(_rows(_row("humanize_off", 0, 1), _row("humanize_h1", 2, 0)), 1,
                      default="humanize_off")["pick"]["pick"] == "humanize_h1"
    moods = {k: v.mood for k, v in PLAN.vamps.items()}
    met = T.vamp_ranks(_rows(_row("float", 0, 3), _row("i_iv", 1, 1), _row("pulse", 2, 0), _row("step", 1, 0),
                             _row("plagal", 3, 0), _row("drift", 1, 1), _row("rise", 0, 1), _row("backdoor", 0, 2)),
                       moods)
    assert met["borrowed"]["met"] and [p["option"] for p in met["moods"]["wistful"]][-1] == "float"
    tie = T.vamp_ranks(_rows(_row("float", 0, 3), _row("i_iv", 1, 1), _row("pulse", 2, 0), _row("step", 1, 0),
                             _row("plagal", 3, 0), _row("drift", 3, 0), _row("rise", 0, 1)), moods)
    assert not tie["borrowed"]["met"] and not tie["borrowed"]["conditions"]["melancholy"]["met"]
    lvl = T.rain_level(_rows(_row("rain_none", 3, 0), _row("rain_d24", 0, 1), _row("rain_d18", 1, 1)), 1,
                       default="rain_d24")
    assert lvl["pick"]["pick"] == "rain_none" and lvl["rain_none_leads"]
    ex = T.rain_excerpt(_rows(_row("excerpt_e1", 1, 0), _row("excerpt_e2", 1, 0), _row("excerpt_e3", 0, 2)), 1,
                        default="excerpt_e1")
    assert ex["pick"]["pick"] == "excerpt_e1" and not ex["pick"]["needs_word"]


def test_day2_and_day3_votes_apply_the_later_rules(inbox, tmp_path):
    votes = tmp_path / "votes"
    vote_days(inbox, votes, DAY1 + DAY2 + DAY3)
    doc, _ = tally(inbox, votes)
    R = doc["rules"]
    assert {"K2", "S1f", "S2a", "S2b", "R3a", "R3b", "V", "W1", "W2"} <= set(R)
    assert R["K2"]["survivors"][0] == "fp_fm_ep1" and set(R["K2"]["survivors"]) <= set(DAY2_LEADERS["keys"])
    assert R["S1f"]["winner"] == "sax_mtg_tenor" and R["S1f"]["rerun_s2"] is None
    assert R["R3a"]["pick"] == "swing_55" and any(n["stage"] == "R3a" for n in doc["needs_word"])
    assert set(R["V"]["moods"]) == {"hopeful", "melancholy", "wistful"}
    assert doc["pooled"]["K1 + K2"]["places"] and doc["pooled"]["S1 + S1f"]["places"]
    md = T.report_md(doc)
    for words in ("Keys survivors (CD-40", "Sax final: winner", "Swing (advisory", "Vamps (seed data)",
                  "Rain excerpt", "Rain level", "### K1 + K2 (pooled", "Repeats (CD-47):"):
        assert words in md, words


# ---------------------------------------------------------------------------------------------- flags and the CLI
def test_rounds_whose_levels_spread_too_far_are_flagged(inbox, tmp_path):
    import shutil
    root = tmp_path / "inbox"
    for n in ("b1103-1-keys", "b1104-1-groove", "b1104-4-vamps"):
        src = inbox["folders"][n]
        shutil.copytree(src, root / src.parent.name / n)
    edits = {}
    for n, stage, field, delta in (("b1103-1-keys", "K1", "bus_lufs", 1.2),
                                   ("b1104-1-groove", "R3a", "bus_lufs", 0.6),
                                   ("b1104-1-groove", "R3b", "bus_lufs", 0.4),
                                   ("b1104-4-vamps", "V", "bass_lufs", 1.1)):
        path = next(root.glob(f"*/{n}")) / HIDDEN_NAME
        hid = json.loads(path.read_text())
        rid, hr = next((k, v) for k, v in hid["rounds"].items()
                       if next(iter(v["clips"].values()))["stage"] == stage and v["kind"] == "none")
        first = next(iter(hr["clips"].values()))
        first[field] = round(first[field] + delta, 4)
        path.write_text(json.dumps(hid, indent=2))
        edits[(stage, field)] = (rid, delta)
    votes = tmp_path / "votes"
    votes.mkdir()
    doc, _ = T.tally(votes, root, plan=PLAN)
    flagged = {(f["stage"], f["field"]): f for f in doc["flags"]}
    assert set(flagged) == {("K1", "bus_lufs"), ("R3a", "bus_lufs"), ("V", "bass_lufs")}   # R3b's 0.4 < 0.5
    assert flagged[("R3a", "bus_lufs")]["limit_lu"] == 0.5 and flagged[("K1", "bus_lufs")]["limit_lu"] == 1.0
    assert flagged[("K1", "bus_lufs")]["spread_lu"] == pytest.approx(1.2) and flagged[("K1", "bus_lufs")]["round"] == \
        edits[("K1", "bus_lufs")][0]
    md = T.report_md(doc)
    assert "3 flag(s) in batches still being voted" in md and "spread 1.2 LU" not in md     # no round named yet
    vote(next(root.glob("*/b1103-1-keys")), votes, Clock(), prefer())
    md2 = T.report_md(T.tally(votes, root, plan=PLAN)[0])
    assert "spread 1.2 LU > 1.0" in md2 and "2 flag(s) in batches still being voted" in md2


def test_day2_rounds_are_flagged_on_every_bus_they_match(inbox, tmp_path):
    """CD-53: a Day-2 clip takes its own fader on every bus its round's varied thing moves, and the batch builder
    writes the stage's bus as `bus_lufs` and each other one as `<bus>_lufs`; the tally checks each of them at the
    stage's limit (0.5 LU for the timing variants, CD-26), and a matched bus with no level is a flag too."""
    import shutil
    root = tmp_path / "inbox"
    src = inbox["folders"]["b1104-1-groove"]
    shutil.copytree(src, root / src.parent.name / src.name)
    path = root / src.parent.name / src.name / HIDDEN_NAME
    hid = json.loads(path.read_text())
    design = [(k, v) for k, v in hid["rounds"].items() if v["kind"] == "none"]
    r3a = [(k, v) for k, v in design if next(iter(v["clips"].values()))["stage"] == "R3a"]
    r3b = [(k, v) for k, v in design if next(iter(v["clips"].values()))["stage"] == "R3b"]
    assert all({"bus_lufs", "sax_lufs", "keys_lufs", "bass_lufs"} <= set(c) for _, v in r3a for c in v["clips"].values())
    assert all({"bus_lufs", "keys_lufs", "bass_lufs"} <= set(c) and "sax_lufs" not in c
               for _, v in r3b for c in v["clips"].values())
    edits = {(r3a[0][0], "sax_lufs"): 0.6, (r3b[0][0], "keys_lufs"): 0.4, (r3b[1][0], "bass_lufs"): 0.7}
    for (rid, field), delta in edits.items():
        first = next(iter(hid["rounds"][rid]["clips"].values()))
        first[field] = round(first[field] + delta, 4)
    next(iter(r3a[1][1]["clips"].values())).pop("keys_lufs")                       # a matched bus with no level
    path.write_text(json.dumps(hid, indent=2))
    (tmp_path / "votes").mkdir()
    doc, _ = T.tally(tmp_path / "votes", root, plan=PLAN)
    got = {(f["round"], f["field"], f["what"], f["limit_lu"], f["bus"]) for f in doc["flags"]}
    assert got == {(r3a[0][0], "sax_lufs", "spread", 0.5, "sax"), (r3b[1][0], "bass_lufs", "spread", 0.5, "bass"),
                   (r3a[1][0], "keys_lufs", "missing", 0.5, "keys")}          # R3b's keys at 0.4 LU stay under 0.5
    largest = {(s["stage"], s["bus"]): s["spread_lu"] for s in doc["largest_spread"]}
    assert largest[("R3a", "sax")] == pytest.approx(0.6) and largest[("R3b", "keys")] == pytest.approx(0.4)
    assert {b for s, b in largest if s == "R3a"} == {"drums", "sax", "keys", "bass"}
    assert {b for s, b in largest if s == "R3b"} == {"drums", "keys", "bass"}


def test_the_cli_writes_private_files_and_refuses_a_public_path(inbox, tmp_path, capsys):
    votes = tmp_path / "votes"
    vote_days(inbox, votes)
    report = tmp_path / "private" / "tally.md"
    rc = T.main(["--votes", str(votes), "--inbox", str(inbox["root"]), "--batch", *DAY1, "--report", str(report),
                 "--runs", str(inbox["root"].parent / "runs")])
    out = capsys.readouterr().out
    assert rc == 0 and "day 2 proposal: ready: --stage day2 --leaders keys=fp_fm_ep1+" in out
    assert next(x for x in out.splitlines() if x.startswith("day 2 proposal")).endswith(f" --prior {JOB1}")
    files = sorted(p.name for p in (votes / "audit").iterdir())
    assert files == sorted([f"{n}.json" for n in DAY1] + ["shootouts.json"])
    doc = json.loads((votes / "audit" / "shootouts.json").read_text())
    assert doc["schema"] == T.SCHEMA and doc["proposal"]["day2"]["status"] == "ready"
    assert json.loads((votes / "audit" / "b1103-3-sax.json").read_text())["schema"] == T.BATCH_SCHEMA
    md = report.read_text()
    for head in ("## Batches", "## Stages", "## Flags (section 3)", "## Hidden checks", "## Rules and proposal",
                 "## Needs Shresth's word", "make submit JOB=jobs/shootout.sbatch ARGS='--stage day2 --leaders"):
        assert head in md, head
    assert "sounds" not in md.replace("nothing here says how anything sounds", "")
    assert "Day 3" not in md and "What could give a repeat away" not in md      # no hint of what comes back, or when
    assert T.main(["--votes", str(votes), "--inbox", str(inbox["root"]), "--out", str(ROOT / "gates" / "x")]) == 2
    assert T.main(["--votes", str(votes), "--inbox", str(inbox["root"]),
                   "--report", str(ROOT / "docs" / "tally.md")]) == 2
    assert T.main(["--votes", str(votes), "--inbox", str(inbox["root"]), "--batch", "b9999-nope"]) == 2
    assert "not under data/" in capsys.readouterr().err
    assert not (ROOT / "gates" / "x").exists() and not (ROOT / "docs" / "tally.md").exists()



# ---------------------------------------------------------------------------- D-013 as amended: every attempt
def test_a_mark_on_a_clip_heard_only_in_an_undone_or_abandoned_attempt_counts(inbox, tmp_path):
    """D-013 as amended (Shresth, 2026-10-31): a clip counts as heard once 1 bar has played, over every attempt of its
    round. A best whose clip played only before an undo (a vote taken back) or before an Esc (an abandoned attempt)
    still counts, the redo's own bars being all 0, and the recount reads the same rows as a log with no redo; a mark
    on a clip that never played in any attempt is left out."""
    folder = inbox["folders"]["b1103-1-keys"]
    m, h = load_batch(folder, check_files=False)
    undo, esc, never = (r.id for r in m.rounds[:3])
    base = prefer()

    def choose(r, hr):
        p = base(r, hr)
        if r.id in (undo, esc):          # the first attempt played every clip 1-2 bars before it was undone or left
            p["heard"] = {c: "11000000" for c in r.ids}
        if r.id == never:                # its best never played: picked on the tile without a bar heard
            p["heard"] = {p["best"]: "00000000"}
        return p

    plain, redone = tmp_path / "plain", tmp_path / "redone"
    plain.mkdir()
    redone.mkdir()
    vote(folder, plain, Clock(), lambda r, hr: {k: v for k, v in choose(r, hr).items() if r.id != undo or k != "heard"})
    vote(folder, redone, Clock(), choose, redo={undo: "undo", esc: "abandon"})
    a, _ = tally(inbox, plain, batches=["b1103-1-keys"])
    b, bd = tally(inbox, redone, batches=["b1103-1-keys"])
    assert a["stages"]["K1"]["options"] == b["stages"]["K1"]["options"]           # the redo changes nothing
    assert b["stages"]["K1"]["unheard_marks"] == 1                                 # the clip that never played
    rows = {r["round"]: r for r in bd["b1103-1-keys"]["votes"]}
    assert all(c["heard"] and c["heard_bars"] == "11000000" and c["own_bars"] == "00000000"
               for c in rows[undo]["clips"])
    assert all(c["heard"] and c["own_bars"] == "00000000" for c in rows[esc]["clips"])
    best = next(c for c in rows[never]["clips"] if c["option"] == rows[never]["best"])
    assert not best["heard"] and best["heard_bars"] == "00000000" and "own_bars" not in best
    audit = bd["b1103-1-keys"]["audit"]
    assert audit["retracted"] == 1 and audit["abandoned"] == 1 and audit["heard"]["heard"] == 23   # merged there too
    # what makes them count is the earlier attempts: on the standing votes' own bars, 9 clips never played
    own = active_votes(read_all(redone)["b1103-1-keys"])
    assert sum(c["heard_bars"] == "00000000" for v in own.values() for c in v["clips"]) == 4 + 4 + 1


def test_repeat_agreement_reads_the_bars_of_every_attempt_of_both_showings(inbox, tmp_path):
    """CD-47 with D-013 as amended: a repeat compares the implied pairs of the clips heard in any attempt of each
    showing, as ui.audit does, so a first showing redone after an undo still gives its pairs."""
    reps = []
    for n in DAY3:
        m, h = load_batch(inbox["folders"][n], check_files=False)
        reps += [(n, r.id, h.rounds[r.id].of) for r in m.rounds if h.rounds[r.id].kind == "repeat"]
    day1 = {r.id: n for n in DAY1 for r in load_batch(inbox["folders"][n], check_files=False)[0].rounds}
    batch, rid, of = next((n, r, o) for n, r, o in reps if o in day1)
    votes = tmp_path / "votes"
    votes.mkdir()
    clock = Clock()
    for n in DAY1:
        vote(inbox["folders"][n], votes, clock, prefer(), redo={of: "undo"} if n == day1[of] else None)
    for n in DAY2 + DAY3:
        vote(inbox["folders"][n], votes, clock, prefer())
    doc, _ = tally(inbox, votes)
    row = next(r for r in doc["repeats"]["rows"] if r["round"] == rid)
    assert row["found"] and row["compared"] > 0                       # the redo's own bars are all 0
    logs = read_all(votes)
    merged = {k: with_merged_heard(v, heard_over_attempts(logs[b])) for b in logs
              for k, v in active_votes(logs[b]).items()}
    assert (row["agree"], row["compared"]) == agreement("shootout", merged[of], merged[rid], 1)
    assert agreement("shootout", active_votes(logs[day1[of]])[of], merged[rid], 1)[1] == 0     # unmerged: no pair


# ---------------------------------------------------------------------------- CD-54: the bass re-listen, R2r
R2R_ORDER = ["gs_wurli_ep200", "gu_tine_ep", "vcsl_upright_knight", "jrhodes3d", "fp_fm_ep1", "fp_upright_kw",
             "fp_fsbs_jazz_guitar", "gu_organ", "kit_virtuosity", "kit_gu_jazz", "kit_swirly",
             "bass_vsco2_pizz", "bass_fp_finger_yr", "bass_gu_acoustic",
             "sax_mtg_alto", "sax_mtg_tenor", "sax_fp_tenor", "sax_weresax_alto", "sax_gu_alto", "sax_gu_tenor"]


def r2_choice(how: dict, order=R2R_ORDER):
    """`prefer(order)` everywhere but R2, whose round on kit k is answered `how[k]`: "T" (no best, worst the least
    liked), "TB" (no best, no worst), "G" (all bad), or a bass id for the best (worst the least liked)."""
    base = prefer(order)

    def choose(r, hr):
        info = {c: hr.clips[c] for c in r.ids}
        if next(iter(info.values()))["stage"] != "R2":
            return base(r, hr)
        kit = next(iter(info.values()))["kit"]
        p = base(r, hr)
        what = how.get(kit, "prefer")
        if what == "T":
            return {"no_best": True, "worst": p["worst"]}
        if what == "TB":
            return {"no_best": True, "no_worst": True}
        if what == "G":
            return {"all_bad": True}
        if what != "prefer":
            best = next(c for c, i in info.items() if i["option"] == what)
            worst = next(c for c in reversed(sorted(r.ids, key=lambda c: order.index(info[c]["option"]))) if c != best)
            return {"best": best, "worst": worst}
        return p
    return choose


def answers(by_slot: dict | None = None, *, by_kit: dict | None = None, order=R2R_ORDER):
    """`prefer(order)` everywhere but the design rounds named: by slot (`R1.3`, `R2r.1`), or an R2 round by its kit
    (`by_kit`). A spec is "G" (all bad), or (best, worst) with None for T (no best) or B (no worst), and an optional
    third item {option: heard_bars} for those tiles' bars in the attempt."""
    base = prefer(order)

    def choose(r, hr):
        first = hr.clips[r.ids[0]]
        spec = None
        if hr.kind == "none":
            spec = (by_kit or {}).get(first.get("kit")) if first["stage"] == "R2" else (by_slot or {}).get(first["slot"])
        if spec is None:
            return base(r, hr)
        if spec == "G":
            return {"all_bad": True}
        best, worst, *heard = spec
        opt = {hr.clips[c]["option"]: c for c in r.ids}
        p = {"best": opt[best]} if best else {"no_best": True}
        p.update({"worst": opt[worst]} if worst else {"no_worst": True})
        if heard:
            p["heard"] = {opt[o]: bars for o, bars in heard[0].items()}
        return p
    return choose


R2R_LEADERS = parse_leaders("keys=gu_tine_ep+gs_wurli_ep200+vcsl_upright_knight/kit=kit_virtuosity+kit_gu_jazz/"
                            "bass=bass_gu_acoustic/sax=sax_mtg_alto+sax_mtg_tenor")      # decisions 2 and 3's line
V, J, W = "kit_virtuosity", "kit_gu_jazz", "kit_swirly"
VS, FP, GU = "bass_vsco2_pizz", "bass_fp_finger_yr", "bass_gu_acoustic"


def r2r_groove(tmp: Path, votes: Path, by_slot: dict, *, clock: Clock | None = None, leaders=R2R_LEADERS,
               seed: int = 0x596D) -> dict[str, Path]:
    """A Day-2 build with R2r into tmp/inbox (beside any Day 1 there), its groove batch voted `answers(by_slot)`."""
    folders = _day(tmp, "day2", seed, leaders=leaders, adds=["R2r"])
    votes.mkdir(exist_ok=True)
    vote(folders["b1104-1-groove"], votes, clock or Clock(), answers(by_slot))
    return folders


def test_an_undecided_bass_adds_r2r_on_the_surviving_kits_and_keeps_the_default(inbox, tmp_path):
    """CD-54, decisions 2 and 3 of 2026-10-31 in shape: no R2 round has a best (T in two, G in one), so job 2's
    ARGS add R2r, the kit slot carries the two surviving kits (R1 leader first) and the bass slot the default; the
    draw and the job take the line, and the groove batch then opens with R2r's three rounds on those kits."""
    votes = tmp_path / "votes"
    vote_days(inbox, votes, choose=r2_choice({"kit_virtuosity": "T", "kit_gu_jazz": "TB", "kit_swirly": "G"}))
    doc, _ = tally(inbox, votes)
    p = doc["proposal"]["day2"]
    leaders = parse_leaders(p["leaders"])
    assert p["status"] == "ready" and p["add"] == ["R2r"] and not p["blocking"]
    assert leaders["kit"] == ("kit_virtuosity", "kit_gu_jazz") and leaders["bass"] == ("bass_gu_acoustic",)
    assert p["args"] == f"--stage day2 --leaders {p['leaders']} --prior {JOB1} --add R2r"
    rl = doc["rules"]["R2"]["relisten"]
    assert rl["adds"] and rl["why"][0] == "no R2 round has a best" and rl["rounds_with_a_best"] == []
    assert rl["rounds"] == {"kit_virtuosity": ["R2r.1", "R2r.3"], "kit_gu_jazz": ["R2r.2"]} and rl["left_out"] == []
    assert p["check"]["day_band"]["kit"] == "kit_virtuosity" and p["check"]["day_band"]["bass"] == "bass_gu_acoustic"
    assert doc["rules"]["cut"]["adds"] == ["R2r"] and doc["rules"]["cut"]["day2_plan"] == 31.2
    assert any(n["stage"] == "R2" and "Day 2 adds R2r" in n["what"] and not n["blocking"] for n in doc["needs_word"])
    assert not any(n["stage"] == "R2" and "provisional" in n["what"] for n in doc["needs_word"])   # no seeded draw
    assert len(rl["why"]) == 2 and rl["takers"] == ["kit_gu_jazz", "kit_virtuosity"]      # both cases hold here
    md = T.report_md(doc)
    assert "Day 2 adds R2r, the bass re-listen (CD-54)" in md and "--add R2r" in md
    d = draw(PLAN, "day2", seed=11, leaders=leaders, adds=p["add"])
    groove = d["batches"][0]
    assert [r["stage"] for r in groove["rounds"][:3]] == ["R2r"] * 3 and "R2r" not in [
        r["stage"] for r in groove["rounds"][3:]]
    assert {r["slot"]: r["clips"][0]["kit"] for r in groove["rounds"][:3]} == {
        "R2r.1": "kit_virtuosity", "R2r.2": "kit_gu_jazz", "R2r.3": "kit_virtuosity"}
    run = tmp_path / "runs" / "shootout-13640005"
    assert job.main(["--run", str(run), "--inbox", str(tmp_path / "inbox2"), *shlex.split(p["args"]),
                     "--dry-run"]) == 0
    assert json.loads((run / "summary" / "leaders.json").read_text())["add"] == ["R2r"]


def test_a_pooled_r2_tie_without_the_default_adds_r2r_only_when_a_surviving_kit_takes_it(inbox, tmp_path):
    """CD-54's second case: the R1 leader's round has no best, so it takes the pooled R2 leader, and the pooled
    leaders tie without the default: R2r. With a best in every surviving kit's own round the same pooled tie
    decides nothing, so nothing is added and the proposal is as before (one kit, its R2 pairing)."""
    # R1 under R2R_ORDER: kit_virtuosity leads, kit_gu_jazz second, kit_swirly last with BW -1: dropped
    tie = r2_choice({"kit_virtuosity": "T", "kit_gu_jazz": "bass_vsco2_pizz", "kit_swirly": "bass_fp_finger_yr"})
    votes = tmp_path / "votes"
    vote_days(inbox, votes, choose=tie)
    doc, _ = tally(inbox, votes)
    rl = doc["rules"]["R2"]["relisten"]
    assert doc["rules"]["R1"]["survivors"][:2] == ["kit_virtuosity", "kit_gu_jazz"] and doc["rules"]["R1"]["dropped"]
    assert rl["adds"] and rl["takers"] == ["kit_virtuosity"] and "tie without the default" in rl["why"][0]
    assert len(rl["rounds_with_a_best"]) == 2 and doc["proposal"]["day2"]["add"] == ["R2r"]
    decided = r2_choice({"kit_virtuosity": "bass_fp_finger_yr", "kit_gu_jazz": "bass_vsco2_pizz", "kit_swirly": "T"})
    votes2 = tmp_path / "votes2"
    vote_days(inbox, votes2, choose=decided)
    doc2, _ = tally(inbox, votes2)
    p2 = doc2["proposal"]["day2"]
    pooled = doc2["rules"]["R2"]["pooled_leader"]
    assert pooled["needs_word"] and sorted(pooled["tied"]) == ["bass_fp_finger_yr", "bass_vsco2_pizz"]
    assert not doc2["rules"]["R2"]["relisten"]["adds"] and p2["add"] == [] and "--add" not in p2["args"]
    leaders = parse_leaders(p2["leaders"])
    assert leaders["kit"] == ("kit_virtuosity",) and leaders["bass"] == ("bass_fp_finger_yr",)


def test_after_day_2_each_kit_takes_the_best_bass_of_its_own_r2r_rounds(tmp_path):
    """CD-43 as CD-54 reads it: the leader kit (two R2r rounds) takes the CD-39 leader of its own rounds; the other
    kit, whose one round has no best, takes the pooled R2r leader. A tie over a kit's own rounds without the default
    goes to Shresth's word, the default staying meanwhile. (That R2's votes are not pooled in is
    test_after_day_2_the_r2_votes_are_not_pooled_in's.)"""
    folders = _day(tmp_path, "day2", 0x596D, leaders=R2R_LEADERS, adds=["R2r"])
    box = {"root": tmp_path / "inbox", "folders": folders}
    m, h = load_batch(folders["b1104-1-groove"], check_files=False)
    r2r = {h.rounds[r.id].clips[r.ids[0]]["slot"]: r.id for r in m.rounds
           if h.rounds[r.id].clips[r.ids[0]]["stage"] == "R2r"}
    assert [m.rounds[i].id for i in range(3)] == sorted(r2r.values(), key=[r.id for r in m.rounds].index)
    assert sorted(r2r) == ["R2r.1", "R2r.2", "R2r.3"]
    votes = tmp_path / "votes"
    votes.mkdir()
    vote(folders["b1104-1-groove"], votes, Clock(), answers({
        "R2r.1": ("bass_fp_finger_yr", "bass_gu_acoustic"), "R2r.3": ("bass_fp_finger_yr", "bass_vsco2_pizz"),
        "R2r.2": (None, "bass_gu_acoustic")}))
    doc, _ = T.tally(votes, box["root"], plan=PLAN, batches=["b1104-1-groove"])
    assert doc["stages"]["R2r"]["status"] == "complete" and doc["stages"]["R2r"]["ready"]
    r = doc["rules"]["R2r"]
    pv, pg = r["pairings"]["kit_virtuosity"], r["pairings"]["kit_gu_jazz"]
    assert (pv["bass"], pv["from"], pv["slots"]) == ("bass_fp_finger_yr", "its own R2r rounds", ["R2r.1", "R2r.3"])
    assert (pg["bass"], pg["from"]) == ("bass_fp_finger_yr", "the pooled R2r leader") and "no counted best" in pg["why"]
    assert r["pooled_leader"]["pick"] == "bass_fp_finger_yr" and not r["pooled_leader"]["needs_word"]
    assert "Basses after R2r (CD-43, CD-54)" in T.report_md(doc)
    tied = tmp_path / "tied"
    tied.mkdir()
    vote(folders["b1104-1-groove"], tied, Clock(), answers({
        "R2r.1": ("bass_fp_finger_yr", "bass_gu_acoustic"), "R2r.3": ("bass_vsco2_pizz", "bass_gu_acoustic"),
        "R2r.2": ("bass_vsco2_pizz", "bass_fp_finger_yr")}))
    doc2, _ = T.tally(tied, box["root"], plan=PLAN, batches=["b1104-1-groove"])
    pv2 = doc2["rules"]["R2r"]["pairings"]["kit_virtuosity"]
    assert pv2["pick"]["needs_word"] and sorted(pv2["pick"]["tied"]) == ["bass_fp_finger_yr", "bass_vsco2_pizz"]
    assert pv2["bass"] == "bass_gu_acoustic"                               # the default meanwhile (CD-39)
    assert doc2["rules"]["R2r"]["pairings"]["kit_gu_jazz"]["bass"] == "bass_vsco2_pizz"
    assert any(n["stage"] == "R2r" and "kit_virtuosity" in n["what"] for n in doc2["needs_word"])


def test_one_g_in_the_leaders_two_r2r_rounds_still_leaves_it_its_best_bass(tmp_path):
    """CD-44 over every R2r round (SHOOTOUTS.md 7.9): a G in one of the leader kit's two R2r rounds makes every bass
    all-bad in half of that kit's rounds but in one of the three R2r rounds, so every bass can still survive and the
    leader takes the best of its other round; Shresth used G in R2 on Day 1."""
    votes = tmp_path / "votes"
    r2r_groove(tmp_path, votes, {"R2r.1": "G", "R2r.3": (FP, VS), "R2r.2": (VS, GU)})
    doc, _ = T.tally(votes, tmp_path / "inbox", plan=PLAN, batches=["b1104-1-groove"])
    r = doc["rules"]["R2r"]
    assert "raised" not in r and not any(o["cannot_survive"] for o in r["rank"])
    assert doc["stages"]["R2r"]["escape"]["all_bad"] == 1
    pv, pg = r["pairings"][V], r["pairings"][J]
    assert (pv["bass"], pv["from"], pv["best_in"]) == (FP, "its own R2r rounds", ["R2r.3"])
    assert (pg["bass"], pg["from"]) == (VS, "its own R2r rounds")
    assert not any(n["stage"] == "R2r" and n["blocking"] for n in doc["needs_word"])


def test_after_day_2_the_r2_votes_are_not_pooled_in(tmp_path):
    """Section 8 Basses after R2r: R2r decides. Day 1's R2 votes, a best on bass_vsco2_pizz in every round, change
    nothing when the Day-1 batches are tallied with the groove batch: the same pairings and the same pooled leader as
    the groove batch alone, where the leader kit's own R2r rounds tie fp and vsco2 and so keep the default."""
    folders = _day(tmp_path, "day1", 0x596A)
    votes = tmp_path / "votes"
    clock = vote_days({"root": tmp_path / "inbox", "folders": folders}, votes,
                      choose=r2_choice(dict.fromkeys((V, J, W), VS)))
    r2r_groove(tmp_path, votes, {"R2r.1": (FP, GU), "R2r.3": (VS, GU), "R2r.2": (FP, VS)}, clock=clock)
    alone, _ = T.tally(votes, tmp_path / "inbox", plan=PLAN, batches=["b1104-1-groove"])
    both, _ = T.tally(votes, tmp_path / "inbox", plan=PLAN, batches=[*DAY1, "b1104-1-groove"], runs=tmp_path / "runs")
    assert "R2" not in alone["stages"] and both["stages"]["R2"]["voted"] == 3          # R2 is read, every best vsco2
    assert {p["bass"] for p in both["rules"]["R2"]["pairings"].values()} == {VS}
    a, b = alone["rules"]["R2r"], both["rules"]["R2r"]
    assert {k: (p["bass"], p["from"]) for k, p in b["pairings"].items()} == {
        k: (p["bass"], p["from"]) for k, p in a["pairings"].items()}
    assert a["rank"] == b["rank"] and a["pooled_leader"]["pick"] == b["pooled_leader"]["pick"] == FP
    assert a["pairings"][V]["pick"]["needs_word"] and a["pairings"][V]["bass"] == GU   # fp and vsco2 tie on its rounds
    assert a["pairings"][J]["bass"] == FP


def test_an_r2_best_on_a_clip_that_never_played_is_no_best(inbox, tmp_path):
    """Decision 7's "has a best" (CD-38, CD-54): a best on a clip that never played in any attempt is no best. Here
    the only R2 best is on such a clip, so no R2 round has a best and Day 2 adds R2r; that kit takes the pooled R2
    leader for that reason."""
    votes = tmp_path / "votes"
    vote_days(inbox, votes, choose=answers(by_kit={V: (FP, GU, {FP: "00000000"}), J: (None, VS), W: "G"}))
    doc, _ = tally(inbox, votes)
    rl, pv = doc["rules"]["R2"]["relisten"], doc["rules"]["R2"]["pairings"][V]
    assert doc["stages"]["R2"]["unheard_marks"] == 1
    assert rl["adds"] and rl["why"] == ["no R2 round has a best"] and rl["rounds_with_a_best"] == []
    assert (pv["from"], pv["why"]) == ("the pooled R2 leader", "its best never played in any attempt")
    assert doc["proposal"]["day2"]["args"].endswith(f"--prior {JOB1} --add R2r")


def test_an_r2_best_heard_only_before_an_undo_counts(inbox, tmp_path):
    """The same votes, but the best's clip played 1 bar in an attempt that was then undone (D-013 as amended: the
    bars of every attempt count): it is a best, so its kit takes it from its own R2 round and nothing is added."""
    m, h = load_batch(inbox["folders"]["b1103-2-groove"], check_files=False)
    rd = next(r for r in m.rounds if h.rounds[r.id].kind == "none"
              and h.rounds[r.id].clips[r.ids[0]]["stage"] == "R2" and h.rounds[r.id].clips[r.ids[0]]["kit"] == V)
    votes = tmp_path / "votes"
    vote_days(inbox, votes, choose=answers(by_kit={V: (FP, GU, {FP: "10000000"}), J: (None, VS), W: "G"}),
              redo={rd.id: "undo"})
    doc, bd = tally(inbox, votes)
    rl, pv = doc["rules"]["R2"]["relisten"], doc["rules"]["R2"]["pairings"][V]
    best = next(c for c in next(x for x in bd["b1103-2-groove"]["votes"] if x["round"] == rd.id)["clips"]
                if c["option"] == FP)
    assert best["own_bars"] == "00000000" and best["heard_bars"] == "10000000" and best["heard"]
    assert not rl["adds"] and rl["rounds_with_a_best"] == [h.rounds[rd.id].clips[rd.ids[0]]["slot"]]
    assert (pv["bass"], pv["from"]) == (FP, "its own R2 round") and doc["stages"]["R2"]["unheard_marks"] == 0
    assert "--add" not in doc["proposal"]["day2"]["args"]


def test_an_r2r_best_that_never_played_leaves_its_kit_the_pooled_r2r_leader(tmp_path):
    """After Day 2 too, a counted best is a best on a clip that played: the leader kit's only R2r best never played,
    so it takes the pooled R2r leader (vsco2: BW 0 like fp, with more bests)."""
    votes = tmp_path / "votes"
    r2r_groove(tmp_path, votes, {"R2r.1": (FP, GU, {FP: "00000000"}), "R2r.3": (None, VS), "R2r.2": (VS, GU)})
    doc, _ = T.tally(votes, tmp_path / "inbox", plan=PLAN, batches=["b1104-1-groove"])
    r = doc["rules"]["R2r"]
    pv = r["pairings"][V]
    assert doc["stages"]["R2r"]["unheard_marks"] == 1 and r["pooled_leader"]["pick"] == VS
    assert (pv["bass"], pv["from"]) == (VS, "the pooled R2r leader") and "no counted best" in pv["why"]


def test_three_r1_survivors_r2r_plays_the_leader_and_the_next_and_the_third_takes_the_pooled_leader(tmp_path):
    """SHOOTOUTS.md 7.9 (CD-54): with three R1 survivors R2r plays the R1 leader (R2r.1 and R2r.3) and the next in
    R1's order (R2r.2); the third plays no R2r round, a non-blocking item says so, and after Day 2 it takes the
    pooled R2r leader."""
    folders = _day(tmp_path, "day1", 0x596A)
    box = {"root": tmp_path / "inbox", "folders": folders}
    votes = tmp_path / "votes"
    clock = vote_days(box, votes, choose=answers({"R1.1": (V, J), "R1.2": (V, W), "R1.3": (V, None)},
                                                 by_kit=dict.fromkeys((V, J, W), (None, None))))
    doc, _ = T.tally(votes, box["root"], plan=PLAN, runs=tmp_path / "runs")
    r1, rl, p = doc["rules"]["R1"], doc["rules"]["R2"]["relisten"], doc["proposal"]["day2"]
    assert r1["survivors"] == [V, J, W] and r1["tie_for_last"] and r1["dropped"] is None
    assert rl["kits"] == [V, J] and rl["left_out"] == [W] and rl["rounds"] == {V: ["R2r.1", "R2r.3"], J: ["R2r.2"]}
    assert parse_leaders(p["leaders"])["kit"] == (V, J) and p["add"] == ["R2r"]
    assert any(n["stage"] == "R2" and not n["blocking"] and W in n["what"] and "play no R2r round" in n["what"]
               for n in doc["needs_word"])
    f2 = _day(tmp_path, "day2", 0x596E, leaders=parse_leaders(p["leaders"]), adds=p["add"])
    vote(f2["b1104-1-groove"], votes, clock, answers({"R2r.1": (FP, GU), "R2r.3": (FP, VS), "R2r.2": (VS, GU)}))
    d2, _ = T.tally(votes, box["root"], plan=PLAN, batches=[*DAY1, "b1104-1-groove"], runs=tmp_path / "runs")
    r = d2["rules"]["R2r"]
    assert list(r["pairings"]) == [V, J, W] and r["pooled_leader"]["pick"] == FP
    pw = r["pairings"][W]
    assert (pw["bass"], pw["from"], pw["why"], pw["slots"]) == (FP, "the pooled R2r leader", "it plays no R2r round", [])
    assert r["pairings"][V]["bass"] == FP and r["pairings"][J]["bass"] == VS


def test_r2r_plays_the_r1_leader_first_even_when_it_is_not_first_in_r1s_order():
    """CD-54: R2r.1 and R2r.3 go to the R1 leader and R2r.2 to the next survivor in R1's order, wherever the leader
    sits in that order: fully tied survivors are listed by id, and when the default (kit_gu_jazz, first by id) is not
    among them a seeded draw names the provisional leader (CD-39), which need not be the first."""
    undecided = {"pooled_leader": {"needs_word": False, "tied": []}, "pairings": {}}
    for survivors in ([W, V], [W, V, J]):
        rl = T.relisten_bass([], undecided, {"leader": {"pick": V}, "survivors": survivors}, PLAN, default=GU)
        assert rl["adds"] and rl["kits"] == [V, W] and rl["left_out"] == survivors[2:], survivors
        assert rl["rounds"] == {V: ["R2r.1", "R2r.3"], W: ["R2r.2"]} and rl["bass"] == GU


def test_day2s_minutes_are_the_drawn_plans():
    """The cut list reads Day 2's minutes from the plan the draw will play (draw.day_minutes): every design round
    exactly as the draw builds it, each anchor at the mean of the rounds it may imitate; a 4-key K2, a 3-sax final
    and R2r each make Day 2 longer than section 6's fixed 28.3."""
    four = parse_leaders("keys=gu_tine_ep+fp_fm_ep1+gu_organ+jrhodes3d/kit=kit_virtuosity+kit_gu_jazz/"
                         "bass=bass_gu_acoustic/sax=sax_mtg_alto+sax_mtg_tenor+sax_gu_alto")
    for counts, adds in (({}, []), ({"keys": 4, "sax": 3}, []), ({"keys": 4, "sax": 3}, ["R2r"])):
        est = day_minutes(PLAN, 2, counts, (), adds)
        lead = {**four, "keys": four["keys"][:counts.get("keys", 3)], "sax": four["sax"][:counts.get("sax", 2)],
                "kit": four["kit"][:2 if adds else 1]}
        for seed in (1, 2, 3):
            d = draw(PLAN, "day2", seed=seed, leaders=lead, adds=adds)
            for b in d["batches"]:
                design = sum(round_seconds(len(r["clips"]), r["bpm"]) for r in b["rounds"] if r["kind"] == "none")
                pool = [round_seconds(len(r["clips"]), r["bpm"]) for r in b["rounds"]
                        if r["kind"] == "none" and PLAN.stages[r["stage"]].anchor]
                anchors = sum(r["kind"] == "anchor" for r in b["rounds"])
                assert est["batches"][b["name"]]["seconds"] == pytest.approx(design + anchors * sum(pool) / len(pool)
                                                                             if anchors else design)
                assert est["batches"][b["name"]]["rounds"] == len(b["rounds"])
    assert day_minutes(PLAN, 2)["minutes"] == 28.2
    assert day_minutes(PLAN, 2, {"keys": 4})["minutes"] == 28.8 and day_minutes(PLAN, 2, {"sax": 3})["minutes"] == 29.2
    assert day_minutes(PLAN, 2, adds=["R2r"])["batches"]["b1104-1-groove"]["minutes"] == 8.6
    assert T.day2_cut(PLAN, 23.0, adds=["R2r"])["cuts"] == ["vamps-high"]     # at the plan's pace R2r passes 30
    assert T.day2_cut(PLAN, 19.0, adds=["R2r"])["cuts"] == []


# ---------------------------------------------------------------------------- section 2.7: Shresth's word on the cuts
def test_shresths_word_on_day_2s_cuts_overrides_the_cut_list_and_keeps_the_final(inbox, tmp_path, capsys):
    """Section 2.7 with Shresth's word on Day 2's cuts (2026-10-31, "Yes, use these" to no Day-2 cuts). On slow Day-1
    votes the list cuts all three items, the final among them, so the S1 rule names one sax; dropping `--cut` from
    that line by hand leaves one sax id, which the job refuses. Given his word (`day2_cuts`, the CLI's --day2-cuts),
    the S1 rule keeps the final: the line has two saxes and no --cut, the band check and the job take it, and the
    list's reading stays in the report as advice."""
    votes = tmp_path / "votes"
    vote_days(inbox, votes, choose=r2_choice({"kit_virtuosity": "T", "kit_gu_jazz": "TB", "kit_swirly": "G"}),
              commit_ms=92000)                                    # 29.1 min of Day 1: r 1.265
    listed, _ = tally(inbox, votes)
    p0 = listed["proposal"]["day2"]
    assert p0["cut"] == ["vamps-high", "latin3", "final"] and p0["args"].endswith(
        f"--prior {JOB1} --add R2r --cut vamps-high+latin3+final")
    assert parse_leaders(p0["leaders"])["sax"] == ("sax_mtg_alto",) and listed["rules"]["S1"]["final"] is False
    assert listed["rules"]["cut"]["decided"] is None and any(n["stage"] == "cut" for n in listed["needs_word"])
    hand = shlex.split(p0["args"].split(" --cut ")[0])            # --cut dropped by hand: one sax, the final kept
    assert job.main(["--run", str(tmp_path / "runs" / "shootout-13640007"), "--inbox", str(tmp_path / "inbox3"),
                     *hand, "--dry-run"]) == 2
    assert "--leaders sax: 1 ids given, day2 takes [2, 3]" in capsys.readouterr().err

    words = 'Shresth, 2026-10-31, "Yes, use these"'
    doc, _ = tally(inbox, votes, day2_cuts=[], day2_cuts_by=words)
    p, c = doc["proposal"]["day2"], doc["rules"]["cut"]
    assert p["args"] == f"--stage day2 --leaders {p['leaders']} --prior {JOB1} --add R2r" and p["cut"] == []
    assert parse_leaders(p["leaders"])["sax"] == ("sax_mtg_alto", "sax_mtg_tenor")
    assert doc["rules"]["S1"]["final"] is True and p["check"]["day_band"]["sax"] == "sax_mtg_alto"
    assert {k: c[k] for k in ("cuts", "fits", "day2_minutes", "days12_minutes")} == {
        k: listed["rules"]["cut"][k] for k in ("cuts", "fits", "day2_minutes", "days12_minutes")}   # the advice
    assert c["decided"] == {"cuts": [], "by": words, "saves_min": 0.0, "day2_minutes": 39.5, "days12_minutes": 68.6,
                            "fits": False}                        # r x 31.2 and r x (23.0 + 31.2)
    assert not any(n["stage"] == "cut" for n in doc["needs_word"]) and not p["blocking"]
    run = tmp_path / "runs" / "shootout-13640006"
    assert job.main(["--run", str(run), "--inbox", str(tmp_path / "inbox2"), *shlex.split(p["args"]),
                     "--dry-run"]) == 0
    assert json.loads((run / "summary" / "leaders.json").read_text())["cut"] == []
    md = T.report_md(doc)
    assert f"Cuts: none, by {words} (Day 2 at r 39.5 min, Days 1-2 68.6). The list read vamps-high, latin3, final" in md
    assert "- Sax final (CD-18): `sax_mtg_alto`, `sax_mtg_tenor`" in md and "the final is cut" not in md
    # his word may cut too: in the list's order, the S1 rule following it
    some, _ = tally(inbox, votes, day2_cuts=["final", "vamps-high"])
    ps = some["proposal"]["day2"]
    assert ps["cut"] == ["vamps-high", "final"] and ps["args"].endswith("--add R2r --cut vamps-high+final")
    assert parse_leaders(ps["leaders"])["sax"] == ("sax_mtg_alto",) and some["rules"]["cut"]["decided"]["by"] == T.WORD
    assert "`--cut vamps-high+final` (Shresth's word)" in T.report_md(some)
    # the CLI
    cli = ["--votes", str(votes), "--inbox", str(inbox["root"]), "--runs", str(inbox["root"].parent / "runs"),
           "--out", str(tmp_path / "audit")]
    assert T.main(cli + ["--day2-cuts", "none", "--day2-cuts-by", words]) == 0
    out = capsys.readouterr().out
    assert f"day 2 proposal: ready: {p['args']}\n" in out
    assert f"day 2 cuts: none, by {words} (the cut list read vamps-high+latin3+final)" in out
    for bad, said in (("w1-two", "not Day-2 cuts of the plan"), ("latin4", "not Day-2 cuts of the plan"),
                      ("", "not Day-2 cuts of the plan"), ("final+final", "names a cut twice")):
        assert T.main(cli + ["--day2-cuts", bad]) == 2, bad
        assert said in capsys.readouterr().err, bad
    assert T.main(cli + ["--day2-cuts-by", words]) == 2 and "goes with --day2-cuts" in capsys.readouterr().err
    with pytest.raises(T.TallyError, match="not Day-2 cuts"):
        T.tally(votes, inbox["root"], plan=PLAN, day2_cuts=["w2-two"])


# ------------------------------------------------------------------------------ after Day 2: job 3's ARGS (7.9)
SERVE = date(2026, 11, 3)


def facts_days(tmp: Path, *, final_best: str = "sax_mtg_alto") -> dict:
    """Day 1 and a Day 2 with R2r in the shape of the real ones (decisions of 2026-10-31 and 2026-11-01), every vote
    synthetic: Day 1 voted on 2026-10-31 (no R2 round with a best, so R2r on kit_virtuosity + kit_gu_jazz), job 2's
    leaders.json written by the job's own --dry-run, and Day 2 voted on 2026-11-01: R2r gives kit_virtuosity
    bass_vsco2_pizz from its two rounds and kit_gu_jazz bass_fp_finger_yr from its one; the final's best is
    `final_best` in every round; S2a's best is sax_plain, S2b's timing_lag, R3b's humanize_off."""
    s1, s2 = 0x5961, 0x5962
    folders = _day(tmp, "day1", s1)
    folders |= _day(tmp, "day2", s2, leaders=R2R_LEADERS, adds=["R2r"])
    run2 = tmp / "runs" / f"shootout-{s2}"
    args2 = ["--stage", "day2", "--leaders", "keys=gu_tine_ep+gs_wurli_ep200+vcsl_upright_knight/kit=kit_virtuosity+"
             "kit_gu_jazz/bass=bass_gu_acoustic/sax=sax_mtg_alto+sax_mtg_tenor", "--prior", str(s1), "--add", "R2r"]
    assert job.main(["--run", str(run2), "--inbox", str(tmp / "unused"), *args2, "--dry-run"]) == 0
    votes = tmp / "votes"
    votes.mkdir()
    day1 = r2_choice({"kit_virtuosity": "T", "kit_gu_jazz": "TB", "kit_swirly": "G"})
    clock = Clock(datetime(2026, 10, 31, 19, 0, tzinfo=PST))
    for n in DAY1:
        vote(folders[n], votes, clock, day1)
    other = {"sax_mtg_alto": "sax_mtg_tenor", "sax_mtg_tenor": "sax_mtg_alto"}[final_best]
    day2 = answers({"R2r.1": (VS, GU), "R2r.3": (VS, FP), "R2r.2": (FP, VS),
                    **{f"S1f.{i}": (final_best, other) for i in (1, 2, 3)},
                    **{f"S2a.{i}": ("sax_plain", "sax_expr_off") for i in (1, 2, 3)},
                    **{f"S2b.{i}": ("timing_lag", "timing_grid") for i in (1, 2, 3)},
                    **{f"R3b.{i}": ("humanize_off", "humanize_h2") for i in (1, 2, 3)},
                    **{f"R3a.{i}": (None, None) for i in (1, 2)}})
    clock = Clock(datetime(2026, 11, 1, 19, 0, tzinfo=PST))
    for n in DAY2:
        vote(folders[n], votes, clock, day2)
    return {"root": tmp / "inbox", "folders": folders, "votes": votes, "runs": tmp / "runs", "jobs": (str(s1), str(s2))}


def test_job_3s_args_carry_the_day2_picks_both_priors_and_the_repeats_the_serving_date_allows(tmp_path, capsys):
    f = facts_days(tmp_path)
    doc, _ = T.tally(f["votes"], f["root"], plan=PLAN, runs=f["runs"], serve_date=SERVE)
    p = doc["proposal"]["day3"]
    leaders = parse_leaders(p["leaders"])
    assert p["status"] == "ready" and not p["blocking"]
    assert leaders == {"keys": (doc["rules"]["K2"]["leader"]["pick"],), "kit": ("kit_virtuosity",),
                       "bass": ("bass_vsco2_pizz",), "sax": ("sax_mtg_alto",), "swing": ("swing_55",),
                       "humanize": ("humanize_off",), "sax_variant": ("sax_plain",), "timing": ("timing_lag",)}
    assert p["pairings"] == {"kit_virtuosity": "bass_vsco2_pizz", "kit_gu_jazz": "bass_fp_finger_yr"}
    s1, s2 = f["jobs"]
    every = "K1+R1+R2+S1+R2r+R3a+R3b+S1f+S2a+S2b+K2+V"              # plan order: Day 1's, then Day 2's
    assert p["args"] == f"--stage day3 --leaders {p['leaders']} --prior {s1}+{s2} --repeats {every}"
    assert p["rerun_s2"] is None and p["cut"] == [] and p["no_repeat"] == []
    fv = doc["first_votes"]
    assert fv["K1"]["last_vote"] == "2026-10-31" and fv["K1"]["days_before"] == 3 and fv["K1"]["eligible"]
    assert fv["R2r"]["last_vote"] == "2026-11-01" and fv["R2r"]["days_before"] == 2 and fv["R2r"]["eligible"]
    assert all(x["against"] == "2026-11-03" and x["serve_date"] for x in fv.values())
    assert "W1" not in fv, "Day 3's own stages never come back"
    assert p["check"]["day_band"]["timing"] == "timing_lag" and p["check"]["day_band"]["bass"] == "bass_vsco2_pizz"
    # the job takes the line, and the draw makes the five repeats from both priors
    run = tmp_path / "runs" / "shootout-13640101"
    assert job.main(["--run", str(run), "--runs", str(f["runs"]), "--inbox", str(tmp_path / "inbox3"),
                     *shlex.split(p["args"]), "--dry-run"]) == 0
    sent = json.loads((run / "summary" / "leaders.json").read_text())
    assert sent["repeats"] == every.split("+") and sent["prior"] == [s1, s2]
    priors = [load_prior(f["runs"] / f"shootout-{j}" / "summary" / "batches.json") for j in (s1, s2)]
    d3 = draw(PLAN, "day3", seed=5, leaders=leaders, priors=priors, repeats=sent["repeats"])
    reps = [r for b in d3["batches"] for r in b["rounds"] if r["kind"] == "repeat"]
    assert sorted(r["group"] for r in reps) == sorted(PLAN.repeat_groups)
    assert d3["band"]["kit"] == "kit_virtuosity" and d3["band"]["timing"] == "timing_lag"
    md = T.report_md(doc)
    assert f"After Day 2 (job 3, ready): `--leaders {p['leaders']}`" in md and p["submit"] in md
    assert "2026-10-31" not in md, "the vote dates stay in the JSON (RATING.md: he is never told)"
    assert T.main(["--votes", str(f["votes"]), "--inbox", str(f["root"]), "--runs", str(f["runs"]),
                   "--out", str(tmp_path / "audit"), "--serve-date", "2026-11-03"]) == 0
    assert f"day 3 proposal: ready: {p['args']}\n" in capsys.readouterr().out
    assert T.main(["--votes", str(f["votes"]), "--inbox", str(f["root"]), "--out", str(tmp_path / "audit"),
                   "--serve-date", "Nov 3"]) == 2 and "is not a date" in capsys.readouterr().err


def test_a_stage_voted_under_two_days_before_the_serving_date_brings_no_repeat(tmp_path):
    f = facts_days(tmp_path)
    doc, _ = T.tally(f["votes"], f["root"], plan=PLAN, runs=f["runs"], serve_date=date(2026, 11, 2))
    p = doc["proposal"]["day3"]
    assert p["repeats"] == ["K1", "R1", "R2", "S1"] and p["args"].endswith("--repeats K1+R1+R2+S1")
    assert sorted(p["no_repeat"]) == ["sax_variants", "vamps"]          # keys, drums + bass, sax sources from Day 1
    assert doc["first_votes"]["V"]["days_before"] == 1 and not doc["first_votes"]["V"]["eligible"]
    assert any("repeat group vamps" in n["what"] for n in doc["needs_word"])
    none, _ = T.tally(f["votes"], f["root"], plan=PLAN, runs=f["runs"], serve_date=date(2026, 11, 1))
    assert "--repeats" not in none["proposal"]["day3"]["args"] and len(none["proposal"]["day3"]["no_repeat"]) == 5
    late = T.tally(f["votes"], f["root"], plan=PLAN, runs=f["runs"])[0]["proposal"]["day3"]
    assert late["status"] == "needs_serve_date" and late["blocking"] and "--serve-date" in late["blocking"][0]


def test_the_final_picking_the_other_finalist_re_runs_s2_on_day_3(tmp_path):
    f = facts_days(tmp_path, final_best="sax_mtg_tenor")
    doc, _ = T.tally(f["votes"], f["root"], plan=PLAN, runs=f["runs"], serve_date=SERVE)
    p = doc["proposal"]["day3"]
    assert doc["rules"]["S1f"]["winner"] == "sax_mtg_tenor" and p["rerun_s2"] == "sax_mtg_tenor"
    assert parse_leaders(p["leaders"])["sax"] == ("sax_mtg_tenor",) and " --rerun-s2 sax_mtg_tenor" in p["args"]
    assert doc["rules"]["day3_cut"]["rerun_minutes"] > 5.0 and p["status"] == "ready"
    assert job.main(["--run", str(tmp_path / "runs" / "shootout-13640102"), "--runs", str(f["runs"]),
                     "--inbox", str(tmp_path / "inbox3"), *shlex.split(p["args"]), "--dry-run"]) == 0


def test_job_3s_args_wait_for_job_2s_pulled_leaders_and_day_3s_cut_follows_its_minutes(tmp_path):
    f = facts_days(tmp_path)
    (f["runs"] / f"shootout-{f['jobs'][1]}" / "summary" / "leaders.json").unlink()
    doc, _ = T.tally(f["votes"], f["root"], plan=PLAN, runs=f["runs"], serve_date=SERVE)
    assert doc["proposal"]["day3"]["status"] == "blocked"
    assert any(n["stage"] == "day3" and n["blocking"] and "leaders.json" in n["what"] for n in doc["needs_word"])
    # the cut: Day 3 at r x its plan over 30 minutes cuts W1 to two rounds first, then W2; his word overrides
    base = T.day3_cut(PLAN, 60.0, 60.0, 340.0, 0.0)
    assert base["cuts"] == [] and base["fits"] and 14.0 < base["day3_plan"] < 17.0
    slow = T.day3_cut(PLAN, 120.0, 60.0, 340.0, 0.0)                     # r = 2: 2 x 15.x passes 30
    assert slow["cuts"][0] == "w1-two" and slow["saves_min"] > 0
    assert slow["day3_minutes"] <= 30.0 or slow["cuts"] == ["w1-two", "w2-two"]
    word = T.day3_cut(PLAN, 120.0, 60.0, 340.0, 0.0, decided=[], by="his word")
    assert (word["decided"]["cuts"], word["decided"]["by"], word["decided"]["saves_min"]) == ([], "his word", 0.0)
    assert word["decided"]["day3_minutes"] > 30.0 and not word["decided"]["fits"]
    with pytest.raises(T.TallyError, match="not Day-3 cuts"):
        T.tally(f["votes"], f["root"], plan=PLAN, day3_cuts=["latin3"])


def test_a_w1_round_whose_beds_spread_over_1_lu_is_flagged_and_w2s_levels_never_are(inbox, tmp_path):
    import shutil
    root = tmp_path / "inbox"
    for n in DAY3:
        src = inbox["folders"][n]
        shutil.copytree(src, root / src.parent.name / n)
    for n in DAY3:
        path = next(root.glob(f"*/{n}")) / HIDDEN_NAME
        hid = json.loads(path.read_text())
        for hr in hid["rounds"].values():
            first = next(iter(hr["clips"].values()))
            if first["stage"] == "W1" and hr["kind"] == "none":
                first["bed_lu"] = round(first["bed_lu"] + 1.25, 4)
        path.write_text(json.dumps(hid, indent=2))
    votes = tmp_path / "votes"
    votes.mkdir()
    doc, _ = T.tally(votes, root, plan=PLAN)
    flagged = [f for f in doc["flags"] if f["stage"] in ("W1", "W2")]
    assert {f["stage"] for f in flagged} == {"W1"} and len(flagged) == 3
    assert all(f["field"] == "bed_lu" and f["spread_lu"] == pytest.approx(1.25) and f["limit_lu"] == 1.0
               for f in flagged)
    assert "W1" in {s["stage"] for s in doc["largest_spread"]} and "W2" not in {s["stage"] for s in doc["largest_spread"]}
