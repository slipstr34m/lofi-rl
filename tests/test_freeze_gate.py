"""Reading the gate file for the freeze (lofi_rl.freeze.gate; docs/FREEZE.md §4), on synthetic gate texts only.

The real gates/GATE_palette.md keeps changing after the freeze, so these tests never read it (`python -m
lofi_rl.freeze check --gate gates/GATE_palette.md` does, in the laptop checks, job B and the seal; the checks', seal's
and tables' tests run on copies of it). The synthetic text below has the real file's structure and pick items: a moved
or added line passes; a changed pick, a missing label or a missing confirmation fails, naming the heading. Then
Shresth's APPROVED line under `## Approval`, and the reader every test here goes through: a line inside a code fence
or an HTML comment is never read (kit.gates.approvals' rule, LESSONS 52), and the reader keeps one entry per line."""
from __future__ import annotations

import pytest

from lofi_rl.freeze import BED, KEPT, PICKS
from lofi_rl.freeze import gate as G

KEPT_ROWS = "".join(
    f"| {b} | {em} | `{KEPT[b]}` |\n" for b, em in (
        ("d01", "low, melancholy"), ("d02", "low, wistful"), ("d03", "low, hopeful"), ("d04", "mid, melancholy"),
        ("d05", "mid, wistful"), ("d06", "mid, hopeful"), ("d07", "high, melancholy"), ("d08", "high, wistful"),
        ("d09", "high, hopeful"), ("d10", "low, wistful")))

GATE = """\
# Gate: palette_v1 freeze (Phase 1)

After the freeze, `palettes/lofi_rain_v1.yaml` never changes (docs/DECISIONS.md D-017).

## Shootout results

### Day 1: voted 2026-10-31 (batches b1103-1-keys, b1103-2-groove, b1103-3-sax; job 13635646)

**Shresth's confirmation, 2026-10-31:** "Yes, use these". This answered the proposal of the survivors below.
- **The heard rule:** a pick counts once its clip has played 1 bar. His rule: "Imagine it sounded so bad that I did \
not even let it play more than 4 bars."
- **No Day-2 cuts.**

**Survivors**
- **Keys to K2:** `gs_wurli_ep200`, `gu_tine_ep` and `vcsl_upright_knight`, in a full tie. `gu_tine_ep`, the \
default, leads (CD-39). Out: `gu_organ`, `jrhodes3d`, `fp_fm_ep1`, `fp_fsbs_jazz_guitar` and `fp_upright_kw`.
- **Kits:** `kit_virtuosity` and `kit_gu_jazz` survive. `kit_swirly` is dropped (CD-42).
- **Bass:** undecided. The pooled leaders `bass_fp_finger_yr` and `bass_vsco2_pizz` tie without the default.
  - R2r decides it on Day 2 (CD-54).
  - The Day-2 band keeps `bass_gu_acoustic` meanwhile (CD-39).
- **Sax final:** `sax_mtg_alto`, the S1 leader, against `sax_mtg_tenor` (CD-18).

**Day 2's job:** `--stage day2 --prior 13635646`, with no cuts.

K1 (keys):

| place | option | BW |
|---|---|---|
| 1= | `gs_wurli_ep200` | +0.333 |

### Day 2: voted 2026-11-01 (batches b1104-1-groove, b1104-2-sax, b1104-3-keys, b1104-4-vamps; job 13651610)

**Shresth's confirmation, 2026-11-01:** "Yes, use these". This answered the proposal of the results below.

**Results**
- **Sax:** `sax_mtg_alto` wins the final, with more bests (CD-18).
  - Warmth: `sax_plain`. It ties `sax_warm_on` on BW, so the default stays (CD-39).
  - Expression stays on: `sax_expr_off` came last in S2a.
  - Timing: `timing_lag`, the S2b BW leader.
- **Keys:** `gu_tine_ep` (leads), `gs_wurli_ep200` and `vcsl_upright_knight` (CD-40, BW over K1 + K2).
- **Kits and their basses** (CD-43 over R2r, CD-54):
  - `kit_virtuosity` with `bass_vsco2_pizz`, from its two R2r rounds;
  - `kit_gu_jazz` with `bass_fp_finger_yr`, from its one R2r round.
- **Swing:** `swing_55`. R3a had no marks; this is advisory (CD-13), and Shresth named no other.
- **Humanize:** `humanize_off`. R3b had no marks, so the default stays.
- **Vamps** (seed data, CD-45): hopeful `aruarian`, then `royal_half`.
- **Borrowed chords:** the condition is not met, since `float` is first in wistful. `chords.numerals` stays as it is.

**Tables** (the same columns as Day 1; weighted marks):

### Day 3: voted 2026-11-03 (batches b1106-1-mix, b1106-2-mix; job 13656048)

**Shresth's confirmation, 2026-11-03:** "Yes, use these", with "24 below (Recommended)" as his word on the rain level.
He was asked which rain level the palette should use.

**Results**
- **Rain excerpt:** `excerpt_e2`, the W1 BW leader, holding the only best.
- **Rain level:** `rain_d24` (Δ 24 LU under the music), by his word.
  - W2 had no best in any of its 4 rounds.
  - `rain_d24`, `rain_d30` and `rain_none` tied at BW 0.000.
  - `rain_d18` came last with 2 worsts.
- **The sax:** no S2 re-run.

**Tables** (weighted marks):

## Kept drafts (keep3): voted 2026-11-01 (batch b1101-keep3; job 13656049)

The palette's gate item 2: Shresth kept 10 of the 30 drafted loops.

| Brief | Energy, mood | Kept |
|---|---|---|
""" + KEPT_ROWS + """
- **Kept per persona** (descriptive): chords 5, horn 3, groove 2.

## How to approve
Add one line of your own, exactly in this form:

    APPROVED by Shresth Rana <yyyy-mm-dd>

## Approval
"""


def _move(text: str, line: str, before: str) -> str:
    """`text` with the one line `line` taken out and put back just before the one line `before`."""
    assert text.count(line + "\n") == 1 and text.count(before + "\n") == 1, (line, before)
    return text.replace(line + "\n", "", 1).replace(before + "\n", line + "\n" + before + "\n", 1)


def _edit(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def test_the_synthetic_gate_holds_the_picks():
    assert G.picks_problems(GATE) == []
    assert G.kept(GATE) == KEPT and len(KEPT) == 10
    assert G.confirmations(GATE) == {"day1": "2026-10-31", "day2": "2026-11-01", "day3": "2026-11-03"}
    assert G.constants_problems() == []


def test_the_picks_are_what_the_palette_is_built_from():
    assert PICKS["day3"]["items"]["Rain excerpt"][0] == BED["excerpt_from"]
    assert PICKS["day3"]["items"]["Rain level"][0] == BED["delta_from"]
    other = {**PICKS, "day3": {**PICKS["day3"], "items": {**PICKS["day3"]["items"], "Rain excerpt": ("excerpt_e3",)}}}
    assert any("Day 3 bed" in p for p in G.constants_problems(other))


@pytest.mark.parametrize("variant", [
    # an item moved within its list
    lambda t: _move(t, next(ln for ln in t.splitlines() if ln.startswith("- **Kits:**")),
                    next(ln for ln in t.splitlines() if ln.startswith("- **Sax final:**"))),
    # the confirmation moved to the end of its section, below its tables
    lambda t: _move(t, next(ln for ln in t.splitlines() if ln.startswith("**Shresth's confirmation, 2026-11-01")),
                    "### Day 3: voted 2026-11-03 (batches b1106-1-mix, b1106-2-mix; job 13656048)"),
    # keep3's rows in another order, and the whole keep3 section moved above the shootout results
    lambda t: _edit(t, KEPT_ROWS, "".join(reversed(KEPT_ROWS.splitlines(keepends=True)))),
    lambda t: _edit(t.replace(t[t.index("## Kept drafts"):t.index("## How to approve")], ""),
                    "## Shootout results\n", t[t.index("## Kept drafts"):t.index("## How to approve")]
                    + "## Shootout results\n"),
    # lines added: an item and a paragraph, a sign-off section with its counts, the drawn tables after the approval
    lambda t: _edit(t, "- **Vamps** (seed", "- **A note:** `something_new` was asked about.\n- **Vamps** (seed"),
    lambda t: _edit(t, "He was asked which rain level", "A line added later.\n\nHe was asked which rain level"),
    lambda t: _edit(t, "## How to approve", "## Sign-off (b1110-signoff)\n\n- **Heard:** 10 of 10.\n\n"
                                            "## How to approve"),
    lambda t: t + "Entered verbatim from his chat line.\n\n## Drawn round tables (published after the freeze)\n\n"
                  "| pos | slot |\n|---|---|\n| 1 | `K1.3` |\n",
], ids=["item-moved", "confirmation-moved", "rows-reordered", "section-moved", "item-added", "paragraph-added",
        "section-added", "tables-appended"])
def test_a_moved_or_added_line_passes(variant):
    assert G.picks_problems(variant(GATE)) == []


@pytest.mark.parametrize("old, new, heading, what", [
    ("`gs_wurli_ep200` and `vcsl_upright_knight` (CD-40", "`gs_wurli_ep200` and `fp_upright_kw` (CD-40",
     "### Day 2: voted", "**Keys**"),
    ("**Rain excerpt:** `excerpt_e2`", "**Rain excerpt:** `excerpt_e1`", "### Day 3: voted", "**Rain excerpt**"),
    ("Timing: `timing_lag`", "Timing: `timing_swung`", "### Day 2: voted", "Timing"),
    ("`kit_virtuosity` with `bass_vsco2_pizz`", "`kit_virtuosity` with `bass_fp_finger_yr`", "### Day 2: voted",
     "**Kits and their basses**"),
    ("**Sax final:** `sax_mtg_alto`", "**Sax final:** `sax_mtg_tenor`", "### Day 1: voted", "**Sax final**"),
    ("`drafts/v1/d05_groove.ls`", "`drafts/v1/d05_horn.ls`", "## Kept drafts (keep3)", "kept table"),
    ("job 13651610)", "job 13651611)", "### Day 2: voted", "job 13651611"),
])
def test_a_changed_pick_fails_naming_the_heading(old, new, heading, what):
    bad = G.picks_problems(_edit(GATE, old, new))
    assert bad and all(p.startswith(heading) for p in bad), bad
    assert any(what in p for p in bad), bad


@pytest.mark.parametrize("cut, heading, what", [
    ("- **Swing:** `swing_55`. R3a had no marks; this is advisory (CD-13), and Shresth named no other.\n",
     "### Day 2: voted", "no **Swing** item"),
    ("  - Warmth: `sax_plain`. It ties `sax_warm_on` on BW, so the default stays (CD-39).\n", "### Day 2: voted",
     "Warmth"),
    ("- **Rain level:** `rain_d24` (Δ 24 LU under the music), by his word.\n", "### Day 3: voted",
     "no **Rain level** item"),
    ("**Survivors**\n", "### Day 1: voted", "no **Survivors** list"),
])
def test_a_missing_label_fails_naming_the_heading(cut, heading, what):
    bad = G.picks_problems(_edit(GATE, cut, ""))
    assert bad and all(p.startswith(heading) for p in bad), bad
    assert any(what in p for p in bad), bad


@pytest.mark.parametrize("old, new, heading", [
    ('**Shresth\'s confirmation, 2026-10-31:** "Yes, use these". This', "This", "### Day 1: voted"),
    ('"Yes, use these", with "24 below (Recommended)" as', '"Yes, use these" as', "### Day 3: voted"),
    ("**Shresth's confirmation, 2026-11-01:**", "**Shresth's confirmation, 2026-11-02:**", "### Day 2: voted"),
])
def test_a_missing_or_changed_confirmation_fails_naming_the_heading(old, new, heading):
    bad = G.picks_problems(_edit(GATE, old, new))
    assert bad and all(p.startswith(heading) for p in bad), bad
    assert any("confirmation" in p for p in bad), bad


def test_basses_swapped_between_the_kits_is_a_changed_pick():
    # the same ids in another order: each kit plays the other's bass
    swapped = _edit(_edit(GATE, "`kit_virtuosity` with `bass_vsco2_pizz`", "`kit_virtuosity` with `BASS_B`"),
                    "`kit_gu_jazz` with `bass_fp_finger_yr`", "`kit_gu_jazz` with `bass_vsco2_pizz`")
    swapped = swapped.replace("`BASS_B`", "`bass_fp_finger_yr`")
    assert G.picks_problems(swapped) == [
        "### Day 2: voted: **Kits and their basses** holds `kit_virtuosity`, `bass_fp_finger_yr`, `kit_gu_jazz`, "
        "`bass_vsco2_pizz`; the pick is `kit_virtuosity`, `bass_vsco2_pizz`, `kit_gu_jazz`, `bass_fp_finger_yr`"]


def test_a_missing_or_doubled_heading_fails():
    assert G.picks_problems(_edit(GATE, "### Day 2: voted", "### Day 2 voted")) == ["no heading '### Day 2: voted'"]
    doubled = GATE + "\n### Day 3: voted again\n"
    assert any("appears 2 times" in p for p in G.picks_problems(doubled))
    with pytest.raises(G.GateError, match="no heading"):
        G.kept(_edit(GATE, "## Kept drafts (keep3)", "## Kept"))


def test_a_fenced_block_is_never_read():
    fenced = GATE + "\n```\n### Day 1: voted 2026-10-31 (job 13635646)\n**Survivors**\n- **Kits:** `x`\n```\n"
    assert G.picks_problems(fenced) == []


# ------------------------------------------------------------------------------------------- the APPROVED line
def test_no_approval_yet_and_the_example_line_does_not_count():
    assert G.approval_lines(GATE) == ([], []) and G.approval_date(GATE) is None


def test_his_line_is_found_under_the_heading_beside_a_note():
    text = GATE + "\nEntered verbatim from his chat line of 2026-11-05.\nAPPROVED by Shresth Rana 2026-11-05\n"
    assert G.approval_date(text) == "2026-11-05"
    assert G.approval_date(GATE + "APPROVED by Shresth Rana 2026-11-05   \n") == "2026-11-05"
    later = text + "\n## Drawn round tables (published after the freeze)\n\nAPPROVED by Shresth Rana 2026-11-09\n"
    assert G.approval_date(later) == "2026-11-05"


@pytest.mark.parametrize("lines, match", [
    ("APPROVED by Shresth Rana 2026-11-05\nAPPROVED by Shresth Rana 2026-11-06\n", "2 APPROVED lines"),
    ("APPROVED by Shresth Rana 2026-11-5\n", "not the form"),
    ("APPROVED by Shresth Rana 2026-02-30\n", "not the form"),
    ("Approved by Shresth Rana 2026-11-05\n", "not the form"),
    ("APPROVED by Shresth 2026-11-05\n", "not the form"),
    ("    APPROVED by Shresth Rana 2026-11-05\n", "not the form"),
    ("APPROVED by Shresth Rana <2026-11-05>\n", "not the form"),
])
def test_two_lines_or_another_form_refuse(lines, match):
    with pytest.raises(G.GateError, match=match):
        G.approval_date(GATE + lines)


def test_the_approval_heading_is_needed_once():
    with pytest.raises(G.GateError, match="0 '## Approval' headings"):
        G.approval_date(GATE.replace("## Approval\n", ""))
    with pytest.raises(G.GateError, match="2 '## Approval' headings"):
        G.approval_date(GATE + "## Approval\n")


# ------------------------------------------------------------------------------------------- fences and comments
LINE = "APPROVED by Shresth Rana 2026-11-05"


HIDDEN = {
    "comment": f"<!-- the form he writes, for example:\n{LINE}\n-->\n",
    "comment-alone": f"<!--\n{LINE}\n-->\n",
    "comment-after-text": f"A note <!-- that runs on\n{LINE}\n-->\n",
    "one-line-comment": f"<!-- {LINE} -->\n",
    "comment-never-closed": f"<!--\n{LINE}\n",
    "fence": f"```\n{LINE}\n```\n",
    "backticks-in-a-tilde-fence": f"~~~~\n```\n{LINE}\n```\n~~~~\n",
    "shorter-fence-inside": f"````\n```\n{LINE}\n```\n````\n",
    "tildes-in-a-backtick-fence": f"```\n~~~\n{LINE}\n~~~\n```\n",
    "fence-indented-3-with-info": f"   ```yaml\n{LINE}\n   ```\n",
    "fence-never-closed": f"```\n{LINE}\n",
}


@pytest.mark.parametrize("body", list(HIDDEN.values()), ids=list(HIDDEN))
def test_a_line_inside_a_comment_or_a_fence_does_not_count(body):
    assert G.approval_lines(GATE + body) == ([], [])
    assert G.approval_date(GATE + body) is None


@pytest.mark.parametrize("body", [
    f"<!--\nx\n--> <!--\n{LINE}\n-->\n",
    f"<!-- a --> b <!-- c\n{LINE}\n-->\n",
    f"<!--\nx\n--> <!--\n```\n-->\n{LINE}\n```\n",
], ids=["reopened-on-its-closing-line", "opened-again-after-a-close", "reopened-then-a-fence"])
def test_a_comment_opened_after_a_close_on_one_line_refuses(body):
    # kit.gates reads on after the `-->`; a browser hides what follows the second `<!--`. Neither reading is trusted.
    with pytest.raises(G.GateError, match=r"opens an HTML comment after a `-->`"):
        G.approval_date(GATE + body)
    assert len(G._lines(GATE + body)) == len((GATE + body).splitlines())


@pytest.mark.parametrize("body", [f"{LINE}\n", f"A note from chat.\n{LINE}\n", *HIDDEN.values(),
                                  f"<!--\nAPPROVED by Shresth Rana 2026-11-04\n-->\n```\nx\n```\n{LINE}\n"])
def test_the_reader_counts_what_kit_gates_counts(tmp_path, body):
    from kit.gates import approvals
    gate = tmp_path / "GATE_palette.md"
    gate.write_text(GATE + body, encoding="utf-8")
    assert [d for who, d in approvals(gate) if who == "Shresth Rana"] == G.approval_lines(GATE + body)[0]


@pytest.mark.parametrize("body", [
    "<!--\nAPPROVED by Shresth Rana <yyyy-mm-dd>\n-->\n",
    "```\napproved by shresth rana 2026-11-05\n```\n",
    f"<!--\n{LINE}\n{LINE}\n-->\n",
])
def test_an_example_inside_a_comment_or_a_fence_neither_counts_nor_refuses(body):
    assert G.approval_date(GATE + body) is None
    assert G.approval_date(GATE + body + LINE + "\n") == "2026-11-05"


def test_his_line_after_a_closed_comment_and_fence_counts_alone():
    text = GATE + ("<!--\nAPPROVED by Shresth Rana 2026-11-04\n-->\n"
                   "~~~~\n```\nAPPROVED by Shresth Rana 2026-11-03\n```\n~~~~\n" + LINE + "\n")
    assert G.approval_lines(text) == (["2026-11-05"], [])


def test_a_fence_opens_on_at_most_three_spaces():
    # four spaces make an indented block line, not a fence, so his line below it is read
    assert G.approval_date(GATE + f"    ```\n{LINE}\n") == "2026-11-05"


def test_headings_and_items_inside_a_comment_or_a_fence_are_never_read():
    commented = _edit(GATE, "- **Swing:** `swing_55`.", "<!--\n- **Keys:** `x`\n### Day 3: voted again\n-->\n"
                                                        "```\n- **Swing:** `swing_99`\n```\n- **Swing:** `swing_55`.")
    assert G.picks_problems(commented) == []
    hidden = GATE + "<!--\n## Approval\n-->\n```\n## Approval\n## Later\n```\n" + LINE + "\n"
    assert G.approval_date(hidden) == "2026-11-05"
    head = GATE.splitlines().index("## Approval")
    assert G.sections(hidden, G.APPROVAL_HEADING) == [(head, len(hidden.splitlines()))]
    assert G.sections(hidden, "## Later") == []


@pytest.mark.parametrize("text", [GATE, "\ufeff" + GATE, GATE + "<!--\nx\n", GATE + "```\nx\n", "a\r\nb\n", ""])
def test_the_reader_keeps_one_entry_per_line(text):
    assert len(G._lines(text)) == len(text.splitlines())
    if text.lstrip("\ufeff").startswith("# Gate"):
        assert G.picks_problems(text) == []
