"""The Phase-1 gate file (lofi_rl.gate1.gatefile; docs/GATE1.md): its sections in GATE_palette.md's style, the
validity table, the bar's first half with its CI, the readouts, his counts once tallied, kit.gates' How-to-approve
wording with a bare date, an empty `## Approval` that a rewrite keeps byte for byte, and the refusals. Gate files go
to temp folders only; gates/GATE_phase1.md itself is not written here."""
import json
import re
from pathlib import Path

import pytest

from helpers_gate1 import SEED, Clock, copy_run, edit_json, make_run, triage
from kit import gates as KG
from lofi_rl import gate1 as G
from lofi_rl.gate1 import batch as GB
from lofi_rl.gate1 import check as C
from lofi_rl.gate1 import cli
from lofi_rl.gate1 import gatefile as GF
from lofi_rl.ui.manifest import load_batch
from lofi_rl.ui.votes import Ledger

ROOT = Path(__file__).resolve().parents[1]
ARM_B = ROOT / "runs" / "junk-13640325"
DAY = "2026-11-07"
NAME = "b1107-gate1"
HEADINGS = ["# Gate: Phase 1, the base model on the frozen palette", "## The run: `lofi-junk-rate` 13783877",
            "## Validity (D-020's list as amended, with D-022's two swaps)", "## The bar, first half: pass_not_copy",
            "## Beside the bar (D-021: reported, never gated)", "## His ear check", "## Where it stands",
            "## How to approve", "## Approval"]


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """A valid run of 20 clips (clips.json's loudness as made; a stub measure), its gate1 batch, and his votes."""
    root = tmp_path_factory.mktemp("gate1-gate")
    run = root / "junk-13783877"
    info = make_run(run, k=20, measure=False)
    folder = GB.build(G.load_run(run), run / "clips", name=NAME, date=DAY, out_root=root / "inbox", seed=SEED,
                      measure=lambda p: info["measured"][Path(p).name])
    votes = root / "votes"
    votes.mkdir()
    m, h = load_batch(folder, check_files=False)
    led = Ledger(votes, m, h, micro_gap_s=600, ui="test", clock=Clock())
    try:
        for k, r in enumerate(m.rounds):
            led.submit(triage(r, ("great", "good", "fine", "toss")[k % 4]))
    finally:
        led.close()
    return run, folder, votes / f"{NAME}.jsonl"


@pytest.fixture
def run_dir(made, tmp_path) -> Path:
    return copy_run(made[0], tmp_path / "junk-13783877")


def write(run: Path, out: Path, *extra: str, day: str = DAY) -> int:
    return cli.main(["gate", "--run", str(run), "--write", str(out), "--date", day, *extra])


def headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if re.match(r"#{1,6} ", line)]


# --------------------------------------------------------------------------------------------- the file
def test_the_file_has_its_sections_in_order_and_ends_with_an_empty_approval(run_dir, tmp_path, capsys):
    out = tmp_path / "GATE_phase1.md"
    assert write(run_dir, out) == 0
    text = out.read_text()
    assert headings(text) == HEADINGS and text.endswith("\n## Approval\n")
    said = capsys.readouterr().out
    assert f"wrote {out}" in said and "no approval line by Shresth Rana yet" in said
    assert "GATE1_GATE run valid, first half met; ear check not tallied" in said
    assert "- **Not tallied yet.**" in text
    assert "- The first half is met. The second half waits on his ear check." in text


def test_the_head_states_the_bar_and_where_it_comes_from(run_dir, tmp_path):
    out = tmp_path / "GATE_phase1.md"
    write(run_dir, out)
    text = out.read_text()
    assert "has `pass_not_copy.count` at least 300 of 500;" in text
    assert "2. at least 15 of its 50 clips are \"fine\" by Shresth's ear: Fine, Good or Great (TASTE.md, Tiers);" in text
    assert "3. Shresth Rana has written the approval line below." in text
    for source in ("CLAUDE.md §9", "D-019", "D-020 as amended", "D-021 \"What follows\" item 1", "D-022 \"Consequences\""):
        assert source in text
    assert "The bar is D-020's, unchanged." in text


def test_the_run_section_names_the_job_stamp_palette_prompt_and_grammar(run_dir, tmp_path):
    out = tmp_path / "GATE_phase1.md"
    write(run_dir, out)
    text = out.read_text()
    assert "- **Job:** 13783877 on b11-13.hpc.usc.edu (NVIDIA A40, 46068 MiB)." in text
    assert "both ran under the stamp `commit=ac4acfb0d1e2 dirty=0 diff=- at=2026-11-06T19:00:00Z`" in text
    assert f"- **Palette:** `lofi_rain_v1@{G.FROZEN_SHA256[:8]}` (sha256 `{G.FROZEN_SHA256[:8]}…`)." in text
    assert f"- **Prompt:** v2, template `{G.TEMPLATE_SHA256[:8]}…`." in text
    assert f"regex sha256 `{G.REGEX_SHA256[:8]}…` (3,408 characters); backend `xgrammar`" in text
    assert "- **Render step** (recorded, not part of validity, D-020): attempted 20, rendered 20, G7 0" in text


def test_the_validity_table_lists_every_condition_with_what_junk_json_shows(run_dir, tmp_path):
    out = tmp_path / "GATE_phase1.md"
    edit_json(run_dir / "summary" / "junk.json", lambda d: d["prompt"].update(example_mismatch=3))
    write(run_dir, out)
    text = out.read_text()
    rows = [line for line in text.splitlines() if line.startswith("| ") and not line.startswith("| group")]
    names = [c.name for c in C.conditions(json.loads((run_dir / "summary" / "junk.json").read_text()))]
    assert len(rows) == 27 and all(f"`{n}`" in row for n, row in zip(names, rows))
    assert "| As registered | `prompt.example_mismatch` | `3` | `0` | **no** |" in rows
    assert sum(row.endswith("| yes |") for row in rows) == 26
    assert "**Not valid:** 1 of 27 conditions fail (`prompt.example_mismatch`)." in text
    assert "- The run is not valid, so it decides nothing (D-020); the ear check waits for a valid run." in text
    assert f"| As registered | `palette.sha256` (D-022) | `{G.FROZEN_SHA256[:8]}…` | `{G.FROZEN_SHA256[:8]}…` | yes |" in rows


def test_the_bar_and_the_readouts_are_written_as_the_check_gives_them(run_dir, tmp_path):
    out = tmp_path / "GATE_phase1.md"
    write(run_dir, out)
    text = out.read_text()
    doc = C.evaluate(G.load_run(run_dir))
    p, k = doc["readouts"]["passing_non_copies"], doc["readouts"]["clips"]
    assert ("- `pass_not_copy.count` **446** of 500 (rate 0.892; Wilson 95 % CI 0.8617–0.9163, as junk.json gives "
            "it). It needs at least 300: **met**.") in text
    assert (f"- **Among the passing non-copies:** {p['k']} of 446 carry a verbatim A or B line "
            f"({C.pct(p['k'], 446)}); {p['both']} carry both.") in text
    assert f"- **Among the clips he hears:** {k['k']} of 20 ({C.pct(k['k'], 20)}); {k['both']} carry both." in text


def test_how_to_approve_is_kit_gates_wording_with_the_line_written_out_bare(run_dir, tmp_path):
    out = tmp_path / "GATE_phase1.md"
    write(run_dir, out)
    text = out.read_text()
    how = KG.how_to_approve("Shresth Rana", DAY, under="## Approval")
    assert how in text and "    APPROVED by Shresth Rana 2026-11-07\n" in text and "<" not in how
    assert KG.gate_approved(out) is False and KG.attempts(out, "Shresth Rana") == []     # the example never counts


def test_a_rewrite_keeps_everything_under_approval_byte_for_byte(made, run_dir, tmp_path, capsys):
    out = tmp_path / "GATE_phase1.md"
    write(run_dir, out)
    his = "\nAPPROVED by Shresth Rana 2026-11-08\r\nhis note, kept as written  \n\n## Later notes\n- x\n"
    out.write_text(out.read_text() + his, newline="")
    assert KG.gate_approved(out, "Shresth Rana")
    _run, folder, votes = made
    assert write(run_dir, out, "--batch", str(folder), "--votes", str(votes), day="2026-11-08") == 0
    raw = out.read_bytes().decode()
    assert raw.split("\n## Approval\n", 1)[1] == his
    assert "## His ear check: batch b1107-gate1" in raw and "Approving on 2026-11-08, the line is:" in raw
    assert "APPROVED by Shresth Rana" in capsys.readouterr().out
    assert write(run_dir, out, "--batch", str(folder), "--votes", str(votes), day="2026-11-08") == 0
    assert capsys.readouterr().out.startswith(f"{out}: unchanged")
    assert out.read_bytes().decode() == raw


def test_his_counts_and_the_split_once_tallied(made, run_dir, tmp_path):
    _run, folder, votes = made
    out = tmp_path / "GATE_phase1.md"
    assert write(run_dir, out, "--batch", str(folder), "--votes", str(votes)) == 0
    text = out.read_text()
    assert "- **Counts:** 20 of 20 heard; 20 voted and heard. Great 5, Good 5, Fine 5, Toss 5." in text
    assert "- **\"Fine\" (Fine, Good or Great):** 15. It needs at least 15 of 50: **met**." in text
    assert "  - with a verbatim A or B line: \"fine\"" in text and "  - with neither: \"fine\"" in text
    assert "- **Not heard (under 1 bar):** none." in text and "- **Not voted:** none." in text
    assert "- **Audit file** (private): `data/votes/b1107-gate1.jsonl`." in text
    assert "- Both halves of the bar are met. The gate passes with his line under `## Approval`." in text


def test_the_committed_file_holds_no_private_value(made, run_dir, tmp_path):
    """Counts only: no seed, no opaque or real id, no audio hash, no sample index paired with a tier."""
    _run, folder, votes = made
    out = tmp_path / "GATE_phase1.md"
    write(run_dir, out, "--batch", str(folder), "--votes", str(votes))
    text = out.read_text()
    m, h = load_batch(folder, check_files=False)
    secrets = {f"{SEED:016x}"} | {r.id for r in m.rounds} | {r.ids[0] for r in m.rounds}
    for hr in h.rounds.values():
        for info in hr.clips.values():
            secrets |= {info["clip"], info["pcm_sha256"], info["flac_sha256"], info["text_sha256"], info["brief"]}
    assert not [s for s in secrets if s in text]
    assert not re.search(r"\b[0-9a-f]{64}\b", text)


@pytest.mark.parametrize("change, standing", [
    (lambda r: edit_json(r / "summary" / "junk.json", lambda d: d.update(partial=True)), "The run is not valid"),
    (lambda r: (edit_json(r / "summary" / "junk.json", lambda d: d["pass_not_copy"].update(count=299))),
     None),                                                       # cross-checked against scored.jsonl: bad input
])
def test_where_it_stands_follows_the_numbers(run_dir, tmp_path, change, standing):
    change(run_dir)
    out = tmp_path / "GATE_phase1.md"
    if standing is None:
        assert write(run_dir, out) == 2 and not out.exists()
        return
    assert write(run_dir, out) == 0 and f"- {standing}" in out.read_text()


def test_under_the_bar_the_gate_fails_and_goes_to_part_b(tmp_path):
    run = tmp_path / "junk-13783877"
    make_run(run, copies=201, k=2, flacs=False)
    out = tmp_path / "GATE_phase1.md"
    assert write(run, out) == 0
    assert ("- The first half is not met (299 < 300): the gate fails on the frozen palette, and it goes to Shresth "
            "with CLAUDE.md §9's next fallback, D-021's Part B.") in out.read_text()


def test_an_ear_check_under_15_or_incomplete_says_so(made, run_dir, tmp_path):
    _run, folder, _votes = made
    f2 = copy_run(folder, tmp_path / "inbox" / DAY / NAME)
    votes = tmp_path / "votes"
    votes.mkdir()
    m, h = load_batch(f2, check_files=False)
    led = Ledger(votes, m, h, micro_gap_s=600, ui="test", clock=Clock())
    try:
        for r in m.rounds[:18]:
            led.submit(triage(r, "fine" if r.number <= 14 else "toss"))
    finally:
        led.close()
    out = tmp_path / "GATE_phase1.md"
    assert write(run_dir, out, "--batch", str(f2), "--votes", str(votes / f"{NAME}.jsonl")) == 0
    text = out.read_text()
    assert "It needs at least 15 of 50: **not met**, with 2 round(s) still to vote." in text
    assert "- The first half is met. His ear check is not complete: 2 round(s) to vote." in text
    m2 = load_batch(f2, check_files=False)[0]
    led = Ledger(votes, m, h, micro_gap_s=600, ui="test", clock=Clock())
    try:
        for r in m2.rounds[18:]:
            led.submit(triage(r, "toss"))
    finally:
        led.close()
    assert write(run_dir, out, "--batch", str(f2), "--votes", str(votes / f"{NAME}.jsonl")) == 0
    assert ("- The first half is met; the ear check is not (14 \"fine\" < 15): the gate fails on the frozen palette, "
            "and it goes to Shresth with CLAUDE.md §9's next fallback, D-021's Part B.") in out.read_text()


# --------------------------------------------------------------------------------------------- refusals
def test_a_file_without_exactly_one_approval_heading_is_refused(run_dir, tmp_path, capsys):
    out = tmp_path / "GATE_phase1.md"
    for text in ("# Gate: something else\n\nAPPROVED nothing\n", "## Approval\n\n## Approval\n"):
        out.write_text(text)
        assert write(run_dir, out) == 2 and out.read_text() == text
    assert "not a file this writer made" in capsys.readouterr().err
    write_ok = tmp_path / "ok.md"
    write(run_dir, write_ok)
    fenced = write_ok.read_text().replace("## How to approve", "```\n## Approval\n```\n\n## How to approve")
    write_ok.write_text(fenced)                                    # a heading in a code fence is not read
    assert write(run_dir, write_ok) == 0


def test_an_approval_line_above_the_heading_is_refused(run_dir, tmp_path, capsys):
    out = tmp_path / "GATE_phase1.md"
    write(run_dir, out)
    text = out.read_text().replace("\n## Approval\n", "\nAPPROVED by Shresth Rana 2026-11-08\n\n## Approval\n")
    out.write_text(text)
    assert write(run_dir, out) == 2 and out.read_text() == text
    assert "starts as an approval above `## Approval`" in capsys.readouterr().err


def test_batch_and_votes_go_together_and_must_be_the_runs(made, run_dir, tmp_path, capsys):
    _run, folder, votes = made
    out = tmp_path / "GATE_phase1.md"
    assert write(run_dir, out, "--batch", str(folder)) == 2
    assert "--batch and --votes go together" in capsys.readouterr().err
    other = copy_run(made[0], tmp_path / "junk-13999999")
    edit_json(other / "summary" / "junk.json", lambda d: d["generate"]["host"].update(job="13999999"))
    assert write(other, out, "--batch", str(folder), "--votes", str(votes)) == 2
    assert "its clip is from job 13783877" in capsys.readouterr().err and not out.exists()
    fewer = copy_run(made[0], tmp_path / "x" / "junk-13783877")
    edit_json(fewer / "summary" / "clips.json", lambda d: d["clips"].pop())
    assert write(fewer, out, "--batch", str(folder), "--votes", str(votes)) == 2
    assert "the batch must hold each of them exactly once" in capsys.readouterr().err


def test_a_bad_date_or_run_writes_nothing(run_dir, tmp_path):
    out = tmp_path / "GATE_phase1.md"
    assert write(run_dir, out, day="2026-02-30") == 2 and not out.exists()
    assert write(tmp_path / "nowhere", out) == 2 and not out.exists()


# --------------------------------------------------------------------------------------------- arm B
def test_arm_bs_gate_file_shows_the_three_swapped_conditions_failing(tmp_path):
    if not (ARM_B / "summary" / "junk.json").is_file():
        pytest.skip(f"{ARM_B.relative_to(ROOT)}/summary is not on this machine (make pull brings it to the laptop)")
    out = tmp_path / "GATE_armb.md"
    assert write(ARM_B, out) == 0
    text = out.read_text()
    assert ("**Not valid:** 3 of 27 conditions fail (`palette.sha256`, `constrained.regex_sha256`, "
            "`constrained.in_language.regex_sha256`).") in text
    assert "`pass_not_copy.count` **446** of 500" in text and "dev_gu@b2e4a340" in text
