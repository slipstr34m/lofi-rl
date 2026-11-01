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
from lofi_rl.shootout.draw import day_band, draw, load_prior, parse_leaders, write_draw
from lofi_rl.shootout.plan import load_plan
from lofi_rl.ui.audit import agreement
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


def vote(folder: Path, votes: Path, clock: Clock, choose, *, rounds=None, commit_ms: int = 60000) -> None:
    """Vote rounds of one batch through the page's Ledger. `choose(round, hidden round)` gives the picks, and may add
    `heard` ({opaque id: heard_bars}) and `gap` (time since the last event)."""
    m, h = load_batch(folder, check_files=False)
    led = Ledger(votes, m, h, clock=clock)
    try:
        for r in m.rounds if rounds is None else [m.round(x) for x in rounds]:
            p = dict(choose(r, h.rounds[r.id]))
            clock.t += p.pop("gap", timedelta(seconds=90))
            heard = p.pop("heard", {})
            led.submit({"type": "shootout", "round": r.id, **p, "t_ms": {"commit": commit_ms},
                        "clips": [{"clip": c, "pos": i + 1, "heard_bars": heard.get(c, FULL), "heard_ms": 20000}
                                  for i, c in enumerate(r.ids)]})
    finally:
        led.close()


def vote_days(inbox: dict, votes: Path, names=DAY1, choose=None, **kw) -> Clock:
    votes.mkdir(exist_ok=True)
    clock = Clock()
    for n in names:
        vote(inbox["folders"][n], votes, clock, choose or prefer(), **kw)
    return clock


def tally(inbox: dict, votes: Path, **kw):
    return T.tally(votes, inbox["root"], plan=PLAN, today=date(2026, 11, 3), **kw)


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


def test_a_mark_on_a_clip_heard_under_4_bars_is_left_out(inbox, tmp_path):
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
    doc, batches = tally(inbox, votes, batches=["b1103-1-keys"])
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
    originals = {k: v for evs in logs.values() for k, v in active_votes(evs).items()}
    agree = compared = 0
    for row in rep["rows"]:                              # the audit's own figure, pooled
        a, n = agreement("shootout", originals[row["of"]], originals[row["round"]], 4)
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
    assert json.loads((run / "summary" / "leaders.json").read_text())["leaders"] == {
        k: list(v) for k, v in leaders.items()}


def test_the_cut_list_follows_the_day1_minutes_and_the_final_cut_leaves_one_sax(inbox, tmp_path):
    assert T.day2_cut(PLAN, 19.0)["cuts"] == []
    assert T.day2_cut(PLAN, 25.3)["cuts"] == ["vamps-high"]                       # r 1.1: 31.1 min on Day 2
    assert T.day2_cut(PLAN, 27.6)["cuts"] == ["vamps-high", "latin3"]             # r 1.2: Days 1-2 57.1, then 53.5
    c = T.day2_cut(PLAN, 46.0)                                                    # r 2: nothing fits
    assert c["cuts"] == ["vamps-high", "latin3", "final"] and not c["fits"] and c["day2_minutes"] == 39.0
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


def test_the_cli_writes_private_files_and_refuses_a_public_path(inbox, tmp_path, capsys):
    votes = tmp_path / "votes"
    vote_days(inbox, votes)
    report = tmp_path / "private" / "tally.md"
    rc = T.main(["--votes", str(votes), "--inbox", str(inbox["root"]), "--batch", *DAY1, "--report", str(report)])
    out = capsys.readouterr().out
    assert rc == 0 and "day 2 proposal: ready: --stage day2 --leaders keys=fp_fm_ep1+" in out
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
