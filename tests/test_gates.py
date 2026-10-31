"""kit.gates: only a real, dated approval line passes a gate (used for ear-check approvals such as
gates/GATE_palette.md: `APPROVED by Shresth Rana <yyyy-mm-dd>`)."""

import pytest

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
