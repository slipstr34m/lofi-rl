"""docs/GATE1.md states what lofi_rl.gate1 does: its four subcommands, the registered values of its validity table,
the batch's measure tolerance, and arm B's check line as the check prints it (when make pull has brought the run)."""
import re
from pathlib import Path

import pytest

from lofi_rl import gate1 as G
from lofi_rl.gate1 import batch as GB
from lofi_rl.gate1 import check as C

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "GATE1.md").read_text(encoding="utf-8")
ARM_B = ROOT / "runs" / "junk-13640325"


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


def test_arm_bs_line_is_what_the_check_prints():
    if not (ARM_B / "summary" / "junk.json").is_file():
        pytest.skip(f"{ARM_B.relative_to(ROOT)}/summary is not on this machine (make pull brings it to the laptop)")
    text = C.format_text(C.evaluate(G.load_run(ARM_B)))
    line = next(x for x in text.splitlines() if x.startswith("validity: "))
    said = re.sub(r"\s+", " ", DOC.split("**On arm B**", 1)[1].split("exactly", 1)[0])
    assert line.removeprefix("validity: ") in said
