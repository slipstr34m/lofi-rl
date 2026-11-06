"""Shresth's ear check of a gate1 batch, counted (lofi_rl.gate1.tally; docs/GATE1.md): his standing vote per round
(retractions honoured), the tiers, the "fine" count against 15, the split by verbatim line, the clips not heard (under
1 bar over every attempt, D-013 as amended) reported apart, and the refusals of a log that is not the batch's. The
votes are synthetic, written by the rating page's own Ledger into temp folders, never data/votes."""
import json
import shutil
from pathlib import Path

import pytest

from helpers_gate1 import SEED, Clock, abandon, copy_run, make_run, triage
from lofi_rl import gate1 as G
from lofi_rl.gate1 import batch as GB
from lofi_rl.gate1 import cli
from lofi_rl.gate1 import tally as T
from lofi_rl.palette import ROOT
from lofi_rl.ui.manifest import HIDDEN_NAME, load_batch
from lofi_rl.ui.votes import Ledger

NAME, DATE = "b1107-gate1", "2026-11-07"
ALL_HEARD, ONE_BAR, NONE = "11111111", "10000000", "00000000"


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """A gate1 batch of 20 rounds, built by gate1.batch from a synthetic run (a stub measure: clips.json's values)."""
    root = tmp_path_factory.mktemp("gate1-tally")
    run = root / "junk-13783877"
    info = make_run(run, k=20, measure=False)
    folder = GB.build(G.load_run(run), run / "clips", name=NAME, date=DATE, out_root=root / "inbox", seed=SEED,
                      measure=lambda p: info["measured"][Path(p).name])
    return folder


@pytest.fixture
def folder(batch, tmp_path) -> Path:
    return copy_run(batch, tmp_path / "inbox" / DATE / NAME)


@pytest.fixture
def votes(tmp_path) -> Path:
    d = tmp_path / "votes"
    d.mkdir()
    return d


def vote_all(folder: Path, votes: Path, tiers, *, bars=None) -> list[dict]:
    """His votes as the page sends them, one per round in serving order: tiers[k] for round k (None: no vote)."""
    m, h = load_batch(folder, check_files=False)
    led = Ledger(votes, m, h, micro_gap_s=600, ui="test", clock=Clock())
    out = []
    try:
        for k, (r, tier) in enumerate(zip(m.rounds, tiers)):
            if tier is not None:
                out.append(led.submit(triage(r, tier, (bars or {}).get(k, ALL_HEARD))))
    finally:
        led.close()
    return out


def log(votes: Path) -> Path:
    return votes / f"{NAME}.jsonl"


def verbatim_by_round(folder: Path) -> list[bool]:
    m, h = load_batch(folder, check_files=False)
    return [h.rounds[r.id].clips[r.ids[0]]["verbatim_line"] for r in m.rounds]


# --------------------------------------------------------------------------------------------- the counts
TIERS20 = ["great", "good", "fine", "toss"] * 5


def test_every_round_voted_counts_each_tier_and_fine_is_fine_good_or_great(folder, votes):
    vote_all(folder, votes, TIERS20)
    doc = T.tally(folder, log(votes))
    assert (doc["rounds"], doc["voted"], doc["heard"], doc["counted"], doc["complete"]) == (20, 20, 20, 20, True)
    assert doc["tiers"] == {"toss": 5, "fine": 5, "good": 5, "great": 5} and doc["fine_or_better"] == 15
    assert doc["bar"] == {"need": 15, "of": 50, "met": True} and doc["exit"] == 0
    assert doc["not_heard"] == [] and doc["not_voted"] == [] and (doc["retracted"], doc["abandoned"]) == (0, 0)


def test_the_split_follows_the_hidden_verbatim_line(folder, votes):
    vote_all(folder, votes, TIERS20)
    doc = T.tally(folder, log(votes))
    flags = verbatim_by_round(folder)
    assert any(flags) and not all(flags)
    for key, want in (("verbatim_line", True), ("no_verbatim_line", False)):
        tiers = [t for t, f in zip(TIERS20, flags) if f is want]
        x = doc["split"][key]
        assert x["n"] == len(tiers) and x["fine_or_better"] == sum(t != "toss" for t in tiers)
        assert x["tiers"] == {t: tiers.count(t) for t in G.TIERS}
    assert doc["split"]["verbatim_line"]["n"] + doc["split"]["no_verbatim_line"]["n"] == 20


def test_fourteen_fine_does_not_meet_the_bar(folder, votes):
    vote_all(folder, votes, ["fine"] * 14 + ["toss"] * 6)
    doc = T.tally(folder, log(votes))
    assert doc["fine_or_better"] == 14 and doc["bar"]["met"] is False and doc["complete"] and doc["exit"] == 3


def test_an_incomplete_ear_check_exits_3_and_names_the_rounds_not_voted(folder, votes):
    vote_all(folder, votes, ["great"] * 17 + [None] * 3)
    doc = T.tally(folder, log(votes))
    assert doc["fine_or_better"] == 17 and doc["bar"]["met"] and doc["complete"] is False and doc["exit"] == 3
    assert [r["number"] for r in doc["not_voted"]] == [18, 19, 20]
    assert [r["number"] for r in doc["not_heard"]] == [18, 19, 20]   # nothing played in a round with no attempt


def test_a_retracted_vote_no_longer_counts_and_the_next_one_does(folder, votes):
    m, h = load_batch(folder, check_files=False)
    led = Ledger(votes, m, h, micro_gap_s=600, ui="test", clock=Clock())
    try:
        first = led.submit(triage(m.rounds[0], "great"))
        led.submit({"type": "retract", "target": first["event_id"], "why": "undo"})
        led.submit(triage(m.rounds[0], "toss", NONE))                 # heard before the undo: it still counts
        for r in m.rounds[1:]:
            led.submit(triage(r, "fine"))
    finally:
        led.close()
    doc = T.tally(folder, log(votes))
    row = doc["per_round"][0]
    assert (row["tier"], row["heard"], row["bars_heard"]) == ("toss", True, 8)
    assert doc["tiers"] == {"toss": 1, "fine": 19, "good": 0, "great": 0} and doc["retracted"] == 1


def test_heard_is_one_bar_over_every_attempt_and_an_unheard_vote_is_left_out(folder, votes):
    """Round 1: one bar in an abandoned attempt, then a vote with nothing playing: heard (D-013 as amended). Round 2:
    a vote with no bar played in any attempt: not heard, its tier left out of the counts and reported apart."""
    m, h = load_batch(folder, check_files=False)
    led = Ledger(votes, m, h, micro_gap_s=600, ui="test", clock=Clock())
    try:
        led.submit(abandon(m.rounds[0], ONE_BAR))
        led.submit(triage(m.rounds[0], "good", NONE))
        led.submit(triage(m.rounds[1], "great", NONE))
        for r in m.rounds[2:]:
            led.submit(triage(r, "fine", ONE_BAR))
    finally:
        led.close()
    doc = T.tally(folder, log(votes))
    assert doc["per_round"][0]["heard"] is True and doc["per_round"][0]["heard_bars"] == ONE_BAR
    assert doc["per_round"][1]["heard"] is False and doc["per_round"][1]["bars_heard"] == 0
    assert doc["not_heard"] == [{"number": 2, "round": m.rounds[1].id, "sample": doc["per_round"][1]["sample"],
                                 "voted": True, "tier": "great", "bars_heard": 0}]
    assert (doc["voted"], doc["heard"], doc["counted"], doc["complete"]) == (20, 19, 19, True)
    assert doc["tiers"] == {"toss": 0, "fine": 18, "good": 1, "great": 0} and doc["fine_or_better"] == 19
    assert doc["abandoned"] == 1


# --------------------------------------------------------------------------------------------- bad input
def _append(votes: Path, event: dict) -> None:
    with log(votes).open("a") as f:
        f.write(json.dumps(event) + "\n")


def _event(folder: Path, k: int = 0, **over) -> dict:
    m, h = load_batch(folder, check_files=False)
    r = m.rounds[k]
    oid = r.ids[0]
    ev = {"v": 1, "type": "triage", "event_id": "e" + "a" * 24, "ts": "2026-11-07T21:00:00-08:00", "session": "s1107a",
          "micro": 1, "batch": NAME, "round": r.id, "tier": "fine",
          "clips": [{"clip": h.rounds[r.id].real(oid), "oid": oid, "pos": 1, "heard_bars": ALL_HEARD, "heard_ms": 1}]}
    ev.update(over)
    return ev


@pytest.mark.parametrize("what, change, message", [
    ("another batch", lambda f: {"batch": "b1103-signoff"}, "batch 'b1103-signoff', not"),
    ("another mode", lambda f: {"type": "bws4"}, "a bws4 vote in a triage batch"),
    ("a round not in it", lambda f: {"round": "r00000000"}, "round 'r00000000' is not in the batch"),
    ("another round's clip", lambda f: {"clips": _event(f, 1)["clips"]}, "is not round"),
    ("a tier not one of four", lambda f: {"tier": "meh"}, "tier 'meh' is not one of"),
])
def test_a_log_that_is_not_the_batchs_is_bad_input(folder, votes, capsys, what, change, message):
    vote_all(folder, votes, TIERS20)
    _append(votes, _event(folder, 0, **change(folder)))
    with pytest.raises(G.Gate1Error) as err:
        T.tally(folder, log(votes))
    assert any(message in p for p in err.value.problems), err.value.problems
    assert cli.main(["tally", str(folder), "--votes", str(log(votes))]) == 2
    assert "error: bad input, nothing decided" in capsys.readouterr().err


def test_a_missing_log_or_a_folder_that_is_not_a_gate1_batch_is_bad_input(folder, votes, tmp_path):
    with pytest.raises(G.Gate1Error, match="is not a file: the batch's vote log is data/votes/b1107-gate1.jsonl"):
        T.tally(folder, votes / "nothing.jsonl")
    vote_all(folder, votes, TIERS20)
    hid = json.loads((folder / HIDDEN_NAME).read_text())
    hid["category"] = "signoff"
    for hr in hid["rounds"].values():
        hr["category"] = "signoff"
    (folder / HIDDEN_NAME).write_text(json.dumps(hid))
    with pytest.raises(G.Gate1Error, match="hidden category 'signoff', not 'gate1'"):
        T.tally(folder, log(votes))
    with pytest.raises(G.Gate1Error, match="is not a batch folder"):
        T.tally(tmp_path / "nowhere", log(votes))


def test_a_hidden_clip_without_the_gate1_fields_is_bad_input(folder, votes):
    vote_all(folder, votes, TIERS20)
    hid = json.loads((folder / HIDDEN_NAME).read_text())
    for hr in hid["rounds"].values():
        for info in hr["clips"].values():
            del info["verbatim_line"]
    (folder / HIDDEN_NAME).write_text(json.dumps(hid))
    with pytest.raises(G.Gate1Error, match="has no gate1 fields"):
        T.tally(folder, log(votes))


def test_an_unreadable_line_is_counted_never_fatal(folder, votes):
    vote_all(folder, votes, TIERS20)
    with log(votes).open("a") as f:
        f.write('{"type": "triage", "cut by a cra\n')
    doc = T.tally(folder, log(votes))
    assert doc["unreadable_lines"] == 1 and doc["voted"] == 20


def test_the_tally_reads_labels_only(folder, votes):
    """A batch whose FLACs were cleared after voting still tallies (ui.manifest load_batch, check_files=False)."""
    vote_all(folder, votes, TIERS20)
    for p in folder.glob("*.flac"):
        p.unlink()
    assert T.tally(folder, log(votes))["fine_or_better"] == 15


# --------------------------------------------------------------------------------------------- the command
def test_the_command_prints_the_counts_and_the_split(folder, votes, capsys):
    vote_all(folder, votes, TIERS20)
    assert cli.main(["tally", str(folder), "--votes", str(log(votes))]) == 0
    out = capsys.readouterr().out
    assert "rounds 20; voted 20; heard 20; counted 20, the rounds voted and heard\n" in out
    assert "  heard: once 1 bar has played, over every attempt of its round (D-013 as amended)\n" in out
    assert "tiers: Great 5, Good 5, Fine 5, Toss 5" in out
    assert "\"fine\" (Fine, Good or Great): 15; the bar is at least 15 of 50: met" in out
    assert "with a verbatim A or B line: \"fine\"" in out and "with neither line: \"fine\"" in out
    assert "GATE1_TALLY complete, \"fine\" 15, bar met\n" in out
    audit = votes / "audit" / f"{NAME}.gate1-tally.json"        # the JSON beside the votes, as the shootout tally's
    assert out.rstrip().endswith(f"wrote {audit} (private: his per-round tiers)")
    assert json.loads(audit.read_text()) == json.loads(json.dumps(T.tally(folder, log(votes))))


def test_the_json_goes_to_stdout_or_to_a_private_file(folder, votes, tmp_path, capsys):
    vote_all(folder, votes, TIERS20)
    assert cli.main(["tally", str(folder), "--votes", str(log(votes)), "--json", "-"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == T.SCHEMA and len(doc["per_round"]) == 20 and doc["fine_or_better"] == 15
    out = tmp_path / "audit" / "tally.json"                         # outside the repo: allowed
    assert cli.main(["tally", str(folder), "--votes", str(log(votes)), "--json", str(out)]) == 0
    assert json.loads(out.read_text()) == doc
    inside = ROOT / "gates" / "gate1-tally.json"
    assert cli.main(["tally", str(folder), "--votes", str(log(votes)), "--json", str(inside)]) == 2
    assert "his per-round tiers are private (D-010)" in capsys.readouterr().err and not inside.exists()
    assert T.private_path_problem(ROOT / "data" / "votes" / "audit" / "x.json") is None
    assert T.private_path_problem(ROOT / "docs" / "x.json") is not None
