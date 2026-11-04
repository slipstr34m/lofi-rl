"""kit.gates: only a real, dated approval line passes a gate (used for ear-check approvals such as
gates/GATE_palette.md: `APPROVED by Shresth Rana 2026-11-04`, his name and the date as bare year-month-day digits).
The form is shown written out, never as a placeholder, and a line that starts as an approval but is not one is told
what to change."""
import datetime as dt

import pytest

from kit import gates
from kit.gates import approvals, gate_approved
from kit.gates import main as gates_main

PLACEHOLDER = "__" + "HUMAN_NAME" + "__"     # spelled in pieces: bootstrap.sh fills the literal token
INSTRUCTIONS = """# Gate 6: publish
## How to approve
Append `APPROVED by Ada Lovelace <YYYY-MM-DD>` to this file, or say "approve" in chat.
NOT APPROVED yet
"""


def _gate(tmp_path, *lines):
    g = tmp_path / "GATE_6_publish.md"
    g.write_text(INSTRUCTIONS + "".join(line + "\n" for line in lines))
    return g


# ---------------------------------------------------------------- gates
def test_gate_needs_a_real_dated_approval_line(tmp_path):
    """HIGH (first project's review): the gate's own instructions satisfied a substring check."""
    g = _gate(tmp_path)
    assert not gate_approved(g)
    g.write_text(g.read_text() + "   APPROVED by Ada Lovelace 2026-12-09\n")
    assert not gate_approved(g)                                 # indented: not a line of its own
    g.write_text(g.read_text() + "APPROVED by Ada Lovelace 2026-12-09\n")
    assert gate_approved(g) and approvals(g) == [("Ada Lovelace", "2026-12-09")]


@pytest.mark.parametrize("line", [
    "Append `APPROVED by Ada Lovelace <YYYY-MM-DD>` to approve.",
    "APPROVED by <Name> <YYYY-MM-DD>",
    "APPROVED by Ada Lovelace <YYYY-MM-DD>",
    "APPROVED by Ada Lovelace YYYY-MM-DD",
    "  APPROVED by Ada Lovelace 2026-12-09",
    "\tAPPROVED by Ada Lovelace 2026-12-09",
    "> APPROVED by Ada Lovelace 2026-12-09",
    "- APPROVED by Ada Lovelace 2026-12-09",
    "`APPROVED by Ada Lovelace 2026-12-09`",
    "**APPROVED by Ada Lovelace 2026-12-09**",
    "<!-- APPROVED by Ada Lovelace 2026-12-09 -->",
    "APPROVED by Ada Lovelace 12/09/2026",
    "APPROVED by Ada Lovelace 2026-12-9",
    "APPROVED by Ada Lovelace 2026-13-45",
    "APPROVED by Ada Lovelace 2026-02-30",
    "APPROVED by Ada Lovelace 2026-12-09 once the typo is fixed",
    "APPROVED by Ada Lovelace on 2026-12-09?",
    "NOT APPROVED by Ada Lovelace 2026-12-09",
    "approved by Ada Lovelace 2026-12-09",
    "APPROVED by 2026-12-09",
    "APPROVED by Ada Lovelace",
    f"APPROVED by {PLACEHOLDER} 2026-12-09",
    "APPROVED by [Ada Lovelace] 2026-12-09",
])
def test_lines_that_are_not_an_approval(tmp_path, line):
    assert not gate_approved(_gate(tmp_path, line)), line


@pytest.mark.parametrize("line, who", [
    ("APPROVED by Ada Lovelace 2026-12-09", "Ada Lovelace"),
    ("APPROVED by Ada Lovelace 2026-12-09   ", "Ada Lovelace"),
    ("APPROVED by Ada  Lovelace 2026-12-09", "Ada Lovelace"),
    ("APPROVED by shresth 2026-12-09", "shresth"),
    ("APPROVED by Jean-Luc O'Neil Jr. 2026-12-09", "Jean-Luc O'Neil Jr."),
])
def test_lines_that_are_an_approval(tmp_path, line, who):
    g = _gate(tmp_path, line)
    assert gate_approved(g) and gate_approved(g, who)


def test_crlf_and_bom_files(tmp_path):
    g = tmp_path / "g.md"
    g.write_bytes("﻿APPROVED by Ada Lovelace 2026-12-09\r\nmore\r\n".encode("utf-8"))
    assert gate_approved(g)


def test_examples_in_code_fences_and_comments_do_not_count(tmp_path):
    g = _gate(tmp_path, "```", "APPROVED by Ada Lovelace 2026-12-09", "```",
              "~~~~", "APPROVED by Ada Lovelace 2026-12-10", "```", "a ``` line does not close a ~~~~ fence", "~~~~",
              "<!--", "APPROVED by Ada Lovelace 2026-12-11", "-->")
    assert not gate_approved(g)
    g.write_text(g.read_text() + "APPROVED by Ada Lovelace 2026-12-12\n")
    assert approvals(g) == [("Ada Lovelace", "2026-12-12")]


def test_a_named_approver_must_match(tmp_path):
    g = _gate(tmp_path, "APPROVED by Ada Lovelace 2026-12-09")
    assert gate_approved(g, "Ada Lovelace") and gate_approved(g, " Ada   Lovelace ")
    assert not gate_approved(g, "Grace Hopper") and not gate_approved(g, "Ada")
    assert not gate_approved(tmp_path / "missing.md")


def test_gates_cli(tmp_path, capsys):
    g = _gate(tmp_path)
    assert gates_main([str(g)]) == 1 and "NOT approved" in capsys.readouterr().out
    g.write_text(g.read_text() + "APPROVED by Ada Lovelace 2026-12-09\n")
    assert gates_main([str(g), "--name", "Ada Lovelace"]) == 0
    assert "Ada Lovelace 2026-12-09" in capsys.readouterr().out
    assert gates_main([str(g), "--name", "Grace Hopper"]) == 1


# ---------------------------------------------------------------- showing the form, and what to change
# Shresth's first save of his palette line, 2026-11-04: `APPROVED by Shresth Rana <2026-11-04>`, the date in the angle
# brackets of the How-to-approve placeholder `<yyyy-mm-dd>`, and the seal's refusal repeated that placeholder. The form
# is now shown written out with a real, bare date, and a near miss is told what to change. The accepted form is the
# one the tests above pin.
TODAY = "2026-12-09"


def test_example_line_is_the_form_written_out_and_counts(tmp_path):
    assert gates.example_line("Ada Lovelace", TODAY) == "APPROVED by Ada Lovelace 2026-12-09"
    assert gates.example_line(" Ada   Lovelace ", dt.date(2026, 12, 9)) == "APPROVED by Ada Lovelace 2026-12-09"
    before = dt.date.today().isoformat()
    line = gates.example_line("Ada Lovelace")
    assert line in {f"APPROVED by Ada Lovelace {d}" for d in (before, dt.date.today().isoformat())}, "today by default"
    assert gate_approved(_gate(tmp_path, gates.example_line("Ada Lovelace", TODAY)), "Ada Lovelace")
    for name, day in (("<Name>", TODAY), ("[Ada Lovelace]", TODAY), ("Ada Lovelace", "2026-02-30"),
                      ("Ada Lovelace", "<2026-12-09>"), ("Ada Lovelace", "yyyy-mm-dd")):
        with pytest.raises(ValueError):
            gates.example_line(name, day)


def test_how_to_approve_shows_a_real_bare_date_and_its_example_never_counts(tmp_path):
    section = gates.how_to_approve("Shresth Rana", "2026-11-08")
    assert section.startswith("## How to approve\n") and "\n    APPROVED by Shresth Rana 2026-11-08\n" in section
    assert "under `## Approval`" in section and "Approving on 2026-11-08" in section
    for placeholder in ("<", ">", "yyyy", "YYYY", "mm-dd", "<date>"):
        assert placeholder not in section, placeholder
    g = tmp_path / "GATE_x.md"
    g.write_text("# Gate x\n\n" + section + "\n## Approval\n")
    assert approvals(g) == [] and gates.attempts(g) == [], "the indented example is no approval and no near miss"
    g.write_text(g.read_text() + "APPROVED by Shresth Rana 2026-11-08\n")
    assert approvals(g) == [("Shresth Rana", "2026-11-08")], "the example, written as a line of its own, counts"
    assert gates.how_to_approve("Ada Lovelace", TODAY, under=None).count("under `") == 0
    with pytest.raises(ValueError):
        gates.how_to_approve("<Name>", TODAY)


@pytest.mark.parametrize("line, name, fixes", [
    ("APPROVED by Shresth Rana <2026-11-04>", "Shresth Rana", ["write the date bare: 2026-11-04, not <2026-11-04>"]),
    ("APPROVED by Ada Lovelace [2026-12-09]", None, ["write the date bare: 2026-12-09, not [2026-12-09]"]),
    ("APPROVED by Ada Lovelace <YYYY-MM-DD>", None, [f"write the day's date in place of <YYYY-MM-DD>, bare: {TODAY}"]),
    ("APPROVED by Ada Lovelace YYYY-MM-DD", None, [f"write the day's date in place of YYYY-MM-DD, bare: {TODAY}"]),
    ("APPROVED by Ada Lovelace <date>", None, [f"write the day's date in place of <date>, bare: {TODAY}"]),
    ("APPROVED by Ada Lovelace 2026-12-9", None, ["write the date as 2026-12-09, not 2026-12-9"]),
    ("APPROVED by Ada Lovelace 2026/12/09", None, ["write the date as 2026-12-09, not 2026/12/09"]),
    ("APPROVED by Ada Lovelace 12/09/2026", None,
     [f"write the date as year-month-day digits joined by hyphens, as {TODAY} is written, not 12/09/2026"]),
    ("APPROVED by Ada Lovelace 2026-02-30", None, ["2026-02-30 is not a date on the calendar"]),
    ("APPROVED by Ada Lovelace 2026-13-45", None, ["2026-13-45 is not a date on the calendar"]),
    ("APPROVED by Ada Lovelace 2026-12-09 once the typo is fixed", None,
     ["end the line with the date: nothing after it (here 'once the typo is fixed')"]),
    ("APPROVED by Ada Lovelace", None, [f"end the line with the date, written bare: {TODAY}"]),
    ("APPROVED by 2026-12-09", None, ["write the name after by"]),
    ("approved by Ada Lovelace 2026-12-09", None, ["write APPROVED in capitals, not approved"]),
    ("  APPROVED by Ada Lovelace 2026-12-09", None,
     ["start the line at its first character: no spaces or tabs before APPROVED"]),
    ("- APPROVED by Ada Lovelace 2026-12-09", None, ["write the line plain, with nothing before APPROVED (here '-')"]),
    ("APPROVED by Ada 2026-12-09", "Ada Lovelace", ["the name is Ada Lovelace, not Ada"]),
    ("APPROVED by [Ada Lovelace] 2026-12-09", None,
     ["write the name plainly: [Ada Lovelace] holds a placeholder or markup character"]),
    ("APPROVED Ada Lovelace 2026-12-09", "Ada Lovelace",
     ["write by and the name after APPROVED: APPROVED by Ada Lovelace"]),
    ("APPROVED by Ada Lovelace\t2026-12-09", None, ["one space between the words, and nothing else on the line"]),
])
def test_a_near_miss_is_told_what_to_change(tmp_path, line, name, fixes):
    assert not gate_approved(_gate(tmp_path, line), name), "the accepted form is unchanged"
    assert gates.fixes(line, name, today=TODAY) == fixes
    why = gates.explain(line, name, today=TODAY)
    assert why.startswith(f"`{line}` is not an approval line: " + "; ".join(fixes)), why
    fixed = gates.corrected(line, name, today=TODAY)
    if fixed is not None:
        assert why.endswith(f" (the line that counts: {fixed})")
        assert gate_approved(_gate(tmp_path, fixed), name), f"{fixed!r} counts"


def test_the_bracketed_date_he_wrote_is_shown_bare():
    line = "APPROVED by Shresth Rana <2026-11-04>"
    assert gates.explain(line, "Shresth Rana", today="2026-11-08") == (
        "`APPROVED by Shresth Rana <2026-11-04>` is not an approval line: write the date bare: 2026-11-04, not "
        "<2026-11-04> (the line that counts: APPROVED by Shresth Rana 2026-11-04)")
    assert gates.corrected("APPROVED by Shresth Rana <yyyy-mm-dd>", "Shresth Rana", today="2026-11-08") == (
        "APPROVED by Shresth Rana 2026-11-08"), "a placeholder date is shown as today's"


@pytest.mark.parametrize("line", ["APPROVED by Ada Lovelace 2026-12-09", "APPROVED by Ada Lovelace 2026-12-09   ",
                                  "APPROVED by Jean-Luc O'Neil Jr. 2026-12-09"])
def test_an_approval_needs_no_fix(line):
    assert gates.fixes(line) == [] and gates.explain(line).endswith("is an approval line")


def test_attempts_names_each_near_miss_by_line_and_skips_examples(tmp_path):
    g = _gate(tmp_path, "    APPROVED by Ada Lovelace <2026-12-09>", "```", "APPROVED by Ada Lovelace <2026-12-10>",
              "```", "<!--", "APPROVED by Ada Lovelace <2026-12-11>", "-->", "APPROVED by Ada Lovelace <2026-12-12>",
              "NOT APPROVED by Ada Lovelace 2026-12-12")
    got = gates.attempts(g, "Ada Lovelace", today=TODAY)
    number = g.read_text().splitlines().index("APPROVED by Ada Lovelace <2026-12-12>") + 1
    assert got == [(number, "APPROVED by Ada Lovelace <2026-12-12>",
                    "`APPROVED by Ada Lovelace <2026-12-12>` is not an approval line: write the date bare: "
                    "2026-12-12, not <2026-12-12> (the line that counts: APPROVED by Ada Lovelace 2026-12-12)")]
    g.write_text(g.read_text() + "APPROVED by Ada Lovelace 2026-12-12\n")
    assert gate_approved(g, "Ada Lovelace") and [n for n, _, _ in gates.attempts(g, "Ada Lovelace")] == [number]
    assert gates.attempts(tmp_path / "missing.md") == []


def test_gates_cli_explains_a_near_miss_and_prints_how_to_approve(tmp_path, capsys):
    g = _gate(tmp_path, "APPROVED by Ada Lovelace <2026-12-09>")
    assert gates_main([str(g), "--name", "Ada Lovelace"]) == 1
    out = capsys.readouterr().out
    assert "NOT approved" in out and "write the date bare: 2026-12-09, not <2026-12-09>" in out
    assert "(the line that counts: APPROVED by Ada Lovelace 2026-12-09)" in out
    assert gates_main(["--how-to-approve", "--name", "Ada Lovelace", "--date", TODAY]) == 0
    assert capsys.readouterr().out == gates.how_to_approve("Ada Lovelace", TODAY)
    for bad in (["--how-to-approve"], ["--how-to-approve", "--name", "<Name>"], []):
        with pytest.raises(SystemExit) as err:
            gates_main(bad)
        assert err.value.code == 2
