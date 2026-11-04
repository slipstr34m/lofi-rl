"""The freeze of `palettes/lofi_rain_v1.yaml` (docs/FREEZE.md; SHOOTOUTS.md §1 "From votes to the freeze", step 4).

    python -m lofi_rl.freeze build|check|vocab        the generator and its check (lofi_rl.freeze.palette)

This module holds the constants every freeze tool reads, and three small readers of them (`kept_rounds`, `kept_now`,
`tossed_kept`); the code is in its submodules: `palette` (build, check, vocab), `gate` (reading gates/GATE_palette.md),
`checks`, `signoff`, `batch`, `seal`, `tables`. After the freeze the file never changes (docs/DECISIONS.md D-017).

- **PICKS** are Shresth's confirmed shootout results as gates/GATE_palette.md records them: for each day its heading,
  its job, his confirmation (date and quoted words) and, for each bold-labelled item of its **Survivors** or
  **Results** list, every backticked id the item holds, its sub-bullets included, in order. `gate.picks_problems`
  compares the gate file with them; the palette check compares the palette with the drafts' constants they name.
- **KEPT** is keep3's table: the brief and the draft file Shresth kept in it.
- **REDOS**: the redo rounds of keep3 (docs/DRAFTS.md §6), in order, each with the loops he kept in its batch;
  **SIGNOFFS**: every sign-off listen, in order, with the kept files he tossed in it (docs/FREEZE.md §5). The kept list
  in effect is `kept_now()`: KEPT with each redo round's picks applied in order. `gate.picks_problems` compares both
  with the gate file, as it does PICKS.
- **BED**: the rain bed takes its excerpt from the W1 pick and only its level from the W2 pick (rain_d24 also sets
  `bed.excerpt: e1`, the excerpt W2 played its levels on, so applying the two variants in order would freeze e1).
- **CHANGES**: Shresth's changes to an input after the shootouts, each with his words (docs/FREEZE.md §3.5). Empty:
  his answers to Q1 and, after the redo round, Q2 (both 2026-11-03) kept the levels he heard, and nothing changes
  without his words.

The three readers take their constants at call time (tests replace a constant with monkeypatch on this module), and
any of them may be given explicitly instead. Sound claims: none.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lofi_rl.palette import ROOT

PALETTE_ID = "lofi_rain_v1"
FROZEN = ROOT / "palettes" / "lofi_rain_v1.yaml"
GATE = ROOT / "gates" / "GATE_palette.md"
RAIN = ROOT / "shootouts" / "rain_excerpts.json"
PROVENANCE_SCHEMA = "lofi-rl/freeze-provenance/1"

# The lofi-shootout jobs of Days 1-3 and the lofi-drafts job of keep3 (gates/GATE_palette.md, each section's heading),
# then the lofi-drafts job of each redo round under its REDOS key (`keep3_redo1`, ...), once he has picked in it.
JOBS: dict[str, int] = {"day1": 13635646, "day2": 13651610, "day3": 13656048, "keep3": 13656049,
                        "keep3_redo1": 13703830}
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

# The redo rounds of keep3, in order (docs/DRAFTS.md §6). An entry is added once Shresth has picked in its batch, from
# his picks there and by his words only, together with its own section in gates/GATE_palette.md under its heading (a
# Brief/Kept table, as keep3's; `gate.picks_problems` compares the two), and its job goes into JOBS under its key:
#   {"key": "keep3_redo<k>", "folder": "drafts/v1/redo<k>", "batch": "b<MMDD>-keep3-redo[<n>]",
#    "job": <its lofi-drafts job id>, "heading": "## Kept drafts (keep3 redo<k>)",
#    "kept": {brief: "drafts/v1/redo<k>/<brief>_<persona>.ls", ...}}
# The batch is the name the job served, as the gate heading gives it: lofi_rl.drafts.job.batch_name counts the redraft
# batches of its own date (b<MMDD>-keep3-redo, then -redo2, ...), so n need not be the round's k.
# `kept` holds only the briefs he kept a loop for: "none acceptable" for a brief leaves it out, and a later round
# redrafts it. redo1: the sign-off listen b1103-signoff reopened d01, d06, d07 and d10 (gates/GATE_palette.md,
# `## Sign-off listen`); he kept one redraft of each in b1103-keep3-redo on 2026-11-03 (`## Kept drafts (keep3
# redo1)`, lofi-drafts job 13703830; his votes: data/votes/b1103-keep3-redo.jsonl).
REDOS: tuple[dict[str, Any], ...] = (
    {"key": "keep3_redo1", "folder": "drafts/v1/redo1", "batch": "b1103-keep3-redo", "job": JOBS["keep3_redo1"],
     "heading": "## Kept drafts (keep3 redo1)",
     "kept": {"d01": "drafts/v1/redo1/d01_groove.ls", "d06": "drafts/v1/redo1/d06_chords.ls",
              "d07": "drafts/v1/redo1/d07_groove.ls", "d10": "drafts/v1/redo1/d10_chords.ls"}},
)

# Every sign-off listen, in order (gates/GATE_palette.md, each `## Sign-off listen` section: its heading's batch and
# job, its **Tossed:** files). `tossed` holds the kept files he marked Toss in it; a listen that passes has `()`.
# A tossed file stays out of the kept list for good: only a redo round's pick of another file replaces it.
SIGNOFFS: tuple[dict[str, Any], ...] = (
    {"batch": "b1103-signoff", "job": 13701026,
     "tossed": ("drafts/v1/d01_chords.ls", "drafts/v1/d06_horn.ls", "drafts/v1/d07_chords.ls",
                "drafts/v1/d10_groove.ls")},
)

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


# --------------------------------------------------------------------------------------------- the kept list
def kept_rounds(kept: Mapping[str, str] | None = None,
                redos: Sequence[Mapping[str, Any]] | None = None) -> dict[str, tuple[str, str]]:
    """{brief: (the file kept now, the key of the round that picked it)}: KEPT, keep3's picks, with each REDOS round's
    picks applied in order, in KEPT's brief order. Nothing is checked here (`gate.records_problems` does that): a pick
    for a brief keep3 never had would be added at the end."""
    now = {b: (f, "keep3") for b, f in (KEPT if kept is None else kept).items()}
    for r in (REDOS if redos is None else redos):
        for b, f in r["kept"].items():
            now[b] = (f, r["key"])
    return now


def kept_now(kept: Mapping[str, str] | None = None,
             redos: Sequence[Mapping[str, Any]] | None = None) -> dict[str, str]:
    """{brief: the file kept now}: KEPT with each REDOS round's picks applied in order, in brief order."""
    return {b: f for b, (f, _key) in kept_rounds(kept, redos).items()}


def tossed_kept(kept: Mapping[str, str] | None = None, redos: Sequence[Mapping[str, Any]] | None = None,
                signoffs: Sequence[Mapping[str, Any]] | None = None) -> list[tuple[str, str, tuple[str, ...]]]:
    """[(brief, file, the sign-off batches that tossed it)] for each file of kept_now() that a SIGNOFFS listen tossed,
    in brief order. A tossed file stays tossed until a redo round's pick of another file replaces it: a later pick of
    that same file keeps it tossed."""
    signoffs = SIGNOFFS if signoffs is None else signoffs
    out = []
    for b, f in kept_now(kept, redos).items():
        batches = tuple(s["batch"] for s in signoffs if f in s["tossed"])
        if batches:
            out.append((b, f, batches))
    return out
