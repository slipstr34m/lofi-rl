"""The three draft writers (docs/DRAFTS.md §2 holds their cards). Each drafts one loop for each brief.

They differ in what they lead with, never in what they may write: every card is inside docs/TASTE.md and the draft
palette's vocabulary, and every draft passes the same checks (lofi_rl.drafts.check). A draft file is
`<brief id>_<persona id>.ls`, e.g. `d04_horn.ls`.
"""
from __future__ import annotations

import re

PERSONAS: tuple[str, ...] = ("groove", "horn", "chords")
TITLES = {
    "groove": "Groove first: the drums and the bass rhythm, then the vamp and the keys, then the sax",
    "horn": "Sax line first: the phrase and its answer, then a band that leaves it room",
    "chords": "Chords and keys first: the vamp, the voicing and the comp, then the groove and the sax",
}
DRAFT_RE = re.compile(r"(?P<brief>d\d{2})_(?P<persona>[a-z]+)\.ls")

if set(TITLES) != set(PERSONAS):
    raise AssertionError("every persona needs a title")


def draft_name(brief_id: str, persona: str) -> str:
    return f"{brief_id}_{persona}.ls"
