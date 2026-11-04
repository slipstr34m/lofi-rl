"""The frozen palette, pinned forever (docs/DECISIONS.md D-017 and D-022; the freeze spec §5.4). Shresth approved it on
2026-11-04 after his second sign-off listen (gates/GATE_palette.md, `## Approval`), and `python -m lofi_rl.freeze.seal`
set `frozen:` to that date. These tests read the frozen bytes and what they parse to, never a layout that may change:
the rebuild from inputs stays in tests/test_freeze_palette.py, and the gate file is read only for his one line, by its
heading. Sound claims: none."""
from __future__ import annotations

import hashlib
from pathlib import Path

from lofi_rl import prompt_v2
from lofi_rl.grammar import loopscript_regex, regex_sha256
from lofi_rl.palette import load_palette

ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILE = ROOT / "palettes" / "lofi_rain_v1.yaml"
GATE = ROOT / "gates" / "GATE_palette.md"

FROZEN_SHA256 = "09f80a644476abe4d14517d0ec4129d69e1a4bc91eb0935c9ffd152411872183"     # B: the sealed file
SIGNOFF_SHA256 = "d3590c8da2e186de2b5a113a585f17646dfec9390f1eb9de569b1bacaa47bc8a"    # A: what job 13706355 rendered
FROZEN_DATE = "2026-11-04"                                                             # his APPROVED line's date
REGEX_SHA256 = "5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558"
REGEX_CHARS = 3408
FROZEN_LINE = f"frozen: '{FROZEN_DATE}'\n"


def test_the_frozen_file_never_changes():
    data = FROZEN_FILE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FROZEN_SHA256, "D-017: the palette never changes after the freeze"


def test_it_parses_as_lofi_rain_v1_frozen_on_his_date():
    pal = load_palette(FROZEN_FILE, verify=False)
    assert pal.id == "lofi_rain_v1"
    assert str(pal.frozen) == FROZEN_DATE


def test_unsealed_it_is_the_file_the_signoff_rendered():
    text = FROZEN_FILE.read_text(encoding="utf-8")
    assert text.count(FROZEN_LINE) == 1
    unsealed = text.replace(FROZEN_LINE, "frozen: null\n")
    assert hashlib.sha256(unsealed.encode("utf-8")).hexdigest() == SIGNOFF_SHA256
    differ = [(a, b) for a, b in zip(text.splitlines(), unsealed.splitlines()) if a != b]
    assert len(text.splitlines()) == len(unsealed.splitlines()) and differ == [(FROZEN_LINE.rstrip("\n"), "frozen: null")]


def test_its_grammar_is_the_run_1_and_phase_1_gate_grammar():
    pal = load_palette(FROZEN_FILE, verify=False)
    assert regex_sha256(pal) == REGEX_SHA256
    assert len(loopscript_regex(pal)) == REGEX_CHARS


def test_prompt_v2_gives_its_nine_references():
    pal = load_palette(FROZEN_FILE, verify=False)
    assert len(prompt_v2.reference_texts(pal, strict=True)) == 9


def test_his_approval_is_on_record():
    lines = GATE.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Approval") + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    assert [ln for ln in lines[start:end] if ln.startswith("APPROVED")] == [f"APPROVED by Shresth Rana {FROZEN_DATE}"]
