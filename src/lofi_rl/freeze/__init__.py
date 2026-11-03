"""The freeze of `palettes/lofi_rain_v1.yaml` (docs/FREEZE.md; SHOOTOUTS.md §1 "From votes to the freeze", step 4).

    python -m lofi_rl.freeze build|check|vocab        the generator and its check (lofi_rl.freeze.palette)

This module holds only the constants every freeze tool reads; the code is in its submodules: `palette` (build, check,
vocab), `gate` (reading gates/GATE_palette.md). After the freeze the file never changes (docs/DECISIONS.md D-017).

- **PICKS** are Shresth's confirmed shootout results as gates/GATE_palette.md records them: for each day its heading,
  its job, his confirmation (date and quoted words) and, for each bold-labelled item of its **Survivors** or
  **Results** list, every backticked id the item holds, its sub-bullets included, in order. `gate.picks_problems`
  compares the gate file with them; the palette check compares the palette with the drafts' constants they name.
- **KEPT** is keep3's table: the brief and the draft file Shresth kept in it.
- **BED**: the rain bed takes its excerpt from the W1 pick and only its level from the W2 pick (rain_d24 also sets
  `bed.excerpt: e1`, the excerpt W2 played its levels on, so applying the two variants in order would freeze e1).
- **CHANGES**: Shresth's changes to an input after the shootouts, each with his words (docs/FREEZE.md §3.5). Empty:
  his answer to Q1 (2026-11-03) kept the levels he heard, and nothing changes without his words.

Sound claims: none.
"""
from __future__ import annotations

from typing import Any

from lofi_rl.palette import ROOT

PALETTE_ID = "lofi_rain_v1"
FROZEN = ROOT / "palettes" / "lofi_rain_v1.yaml"
GATE = ROOT / "gates" / "GATE_palette.md"
RAIN = ROOT / "shootouts" / "rain_excerpts.json"
PROVENANCE_SCHEMA = "lofi-rl/freeze-provenance/1"

# The lofi-shootout jobs of Days 1-3 and the lofi-drafts job of keep3 (gates/GATE_palette.md, each section's heading).
JOBS: dict[str, int] = {"day1": 13635646, "day2": 13651610, "day3": 13656048, "keep3": 13656049}
LEVELS_DAY3 = f"runs/shootout-{JOBS['day3']}/summary/levels.json"     # where `make pull` puts Day 3's levels file

BED: dict[str, str] = {"excerpt_from": "excerpt_e2", "delta_from": "rain_d24"}

# What the frozen palette's vocabulary gives (docs/FREEZE.md §3.4 g, h): the LoopScript regex, as
# lofi_rl.grammar.loopscript_regex writes it, and prompt v2's references under strict=True.
REGEX_CHARS = 3408
REGEX_SHA256 = "5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558"
V2_REFERENCES = 9

KEPT: dict[str, str] = {
    "d01": "drafts/v1/d01_chords.ls", "d02": "drafts/v1/d02_chords.ls", "d03": "drafts/v1/d03_chords.ls",
    "d04": "drafts/v1/d04_horn.ls", "d05": "drafts/v1/d05_groove.ls", "d06": "drafts/v1/d06_horn.ls",
    "d07": "drafts/v1/d07_chords.ls", "d08": "drafts/v1/d08_horn.ls", "d09": "drafts/v1/d09_chords.ls",
    "d10": "drafts/v1/d10_groove.ls",
}

PICKS: dict[str, dict[str, Any]] = {
    "day1": {
        "heading": "### Day 1: voted",
        "job": JOBS["day1"],
        "confirmation": {"date": "2026-10-31", "words": ("Yes, use these",)},
        "list": "Survivors",
        "items": {
            "Keys to K2": ("gs_wurli_ep200", "gu_tine_ep", "vcsl_upright_knight", "gu_tine_ep", "gu_organ", "jrhodes3d",
                           "fp_fm_ep1", "fp_fsbs_jazz_guitar", "fp_upright_kw"),
            "Kits": ("kit_virtuosity", "kit_gu_jazz", "kit_swirly"),
            "Bass": ("bass_fp_finger_yr", "bass_vsco2_pizz", "bass_gu_acoustic"),
            "Sax final": ("sax_mtg_alto", "sax_mtg_tenor"),
        },
        "subs": {},
    },
    "day2": {
        "heading": "### Day 2: voted",
        "job": JOBS["day2"],
        "confirmation": {"date": "2026-11-01", "words": ("Yes, use these",)},
        "list": "Results",
        "items": {
            "Sax": ("sax_mtg_alto", "sax_plain", "sax_warm_on", "sax_expr_off", "timing_lag"),
            "Keys": ("gu_tine_ep", "gs_wurli_ep200", "vcsl_upright_knight"),
            "Kits and their basses": ("kit_virtuosity", "bass_vsco2_pizz", "kit_gu_jazz", "bass_fp_finger_yr"),
            "Swing": ("swing_55",),
            "Humanize": ("humanize_off",),
            "Borrowed chords": ("float", "chords.numerals"),
        },
        # sub-bullets found by their first word, with their own ids
        "subs": {"Sax": {"Warmth": ("sax_plain", "sax_warm_on"), "Expression": ("sax_expr_off",),
                         "Timing": ("timing_lag",)}},
    },
    "day3": {
        "heading": "### Day 3: voted",
        "job": JOBS["day3"],
        "confirmation": {"date": "2026-11-03", "words": ("Yes, use these", "24 below (Recommended)")},
        "list": "Results",
        "items": {
            "Rain excerpt": ("excerpt_e2",),
            "Rain level": ("rain_d24", "rain_d24", "rain_d30", "rain_none", "rain_d18"),
        },
        "subs": {},
    },
    "keep3": {
        "heading": "## Kept drafts (keep3)",
        "job": JOBS["keep3"],
        "kept": KEPT,
    },
}

# Shresth's changes to the build's inputs, each a mapping {rule, by, date, words, ...} (docs/FREEZE.md §3.5). The one
# rule the build knows is `keys_offset_db` (Q1): {rule: keys_offset_db, by: Shresth, date: YYYY-MM-DD, words: "<his
# words, verbatim>", db: {rhodes: x, wurli: y, piano: z}}, added to each keys fader as the exact sum of the decimals.
# Any other change (the bed, another level, a pick) needs its own rule in code, with its check rule and tests, before
# an entry here can name it; the build refuses an unknown rule (docs/FREEZE.md §3.5 lists the bed's sites).
CHANGES: tuple[Any, ...] = ()
