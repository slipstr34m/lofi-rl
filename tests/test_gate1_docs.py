"""docs/GATE1.md states what lofi_rl.gate1 does: its four subcommands, the registered values of its validity table,
the batch's measure tolerance, the on-the-day steps by the gate run's job, and the check lines of arm B and of the
gate's first launch as the check prints them (when make pull has brought those runs)."""
import re
from pathlib import Path

import pytest

from lofi_rl import gate1 as G
from lofi_rl.gate1 import batch as GB
from lofi_rl.gate1 import check as C

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "GATE1.md").read_text(encoding="utf-8")
ARM_B = ROOT / "runs" / "junk-13640325"
FIRST_LAUNCH = ROOT / "runs" / "junk-13783877"   # the gate's first launch (invalid)


def test_a_section_per_subcommand():
    for usage in ("check RUN_DIR [--json]", "batch RUN_DIR --clips DIR [--inbox DIR] [--date YYYY-MM-DD]",
                  "tally BATCH_DIR --votes FILE [--json OUT]",
                  "gate --run RUN_DIR [--batch BATCH_DIR --votes FILE] --write gates/GATE_phase1.md"):
        assert re.search(rf"^## \d\. `{re.escape(usage)}`$", DOC, re.M), usage
        assert usage in G.__doc__


def test_the_validity_table_holds_the_registered_values_and_every_condition():
    table = DOC.split("| group | conditions |", 1)[1].split("\n\n", 1)[0]
    for sha in (G.MODEL_REVISION, G.TEMPLATE_SHA256, G.FROZEN_SHA256, G.REGEX_SHA256):
        assert f"`{sha[:8]}…`" in table
    for c in C.conditions({}):
        last = c.name.rsplit(".", 1)[-1] if c.name.startswith("failures.") else c.name
        assert c.name == C.STAMP_NAME or f"`{c.name}`" in table or f"`{last}`" in table, c.name
    assert len(C.conditions({})) == 27 and "Each of the 27 conditions" in DOC


def test_the_measure_tolerance_is_the_batchs():
    assert f"within {GB.MEASURE_TOL:g} of clips.json's `lufs` and `true_peak_dbtp`" in DOC


def test_on_the_day_names_the_gate_run_by_its_job():
    """§6's commands read `runs/junk-<job>`, the gate run's job, never a launch's own number (review F5: 13783877 was
    invalid), and an invalid run is re-run under D-020's rule before the batch."""
    day = DOC.split("## 6. On the day", 1)[1].split("\n## ", 1)[0]
    steps = "\n".join(line for line in day.splitlines() if re.match(r"(\d\. |   )", line))
    assert "runs/junk-<job>" in steps and not re.search(r"runs/junk-\d", steps)
    assert "re-run under D-020's rule for an invalid arm before step 2" in re.sub(r"\s+", " ", day)


def test_the_first_launchs_line_is_what_the_check_prints():
    if not (FIRST_LAUNCH / "summary" / "junk.json").is_file():
        pytest.skip(f"{FIRST_LAUNCH.relative_to(ROOT)}/summary is not on this machine (make pull brings it)")
    doc = C.evaluate(G.load_run(FIRST_LAUNCH))
    line = next(x for x in C.format_text(doc).splitlines() if x.startswith("validity: "))
    said = re.sub(r"\s+", " ", DOC.split("**The first launch, job 13783877, is not the gate run:**", 1)[1])
    assert line.removeprefix("validity: ") in said.split("`")[1] and doc["exit"] == 3
    j = G.load_run(FIRST_LAUNCH).junk
    assert (j["n"], j["requested"], j["generate"]["stopped"]) == (384, 500, "deadline")
    assert "`n` 384 of 500, generate stopped by its deadline" in said


def test_arm_bs_line_is_what_the_check_prints():
    if not (ARM_B / "summary" / "junk.json").is_file():
        pytest.skip(f"{ARM_B.relative_to(ROOT)}/summary is not on this machine (make pull brings it to the laptop)")
    text = C.format_text(C.evaluate(G.load_run(ARM_B)))
    line = next(x for x in text.splitlines() if x.startswith("validity: "))
    said = re.sub(r"\s+", " ", DOC.split("**On arm B**", 1)[1].split("exactly", 1)[0])
    assert line.removeprefix("validity: ") in said
