"""docs/DRAFTS.md states what the code does: its brief table is drafts/v1/briefs.yaml, its persona cards are the three
persona ids, its vocabulary and chord lists are the draft palette's, and its trims and faders are the palette's."""
import re
from pathlib import Path

from lofi_rl.drafts import palette as P
from lofi_rl.drafts.briefs import load
from lofi_rl.drafts.personas import PERSONAS

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "DRAFTS.md").read_text(encoding="utf-8")


def test_the_brief_table_is_the_briefs_file():
    rows = re.findall(r"^\| (d\d\d) \| (\w+) \| (\w+) \((\w+)\) \| (\d+) \| (\w+ \w+) \|$", DOC, re.M)
    got = {b: (x.brief.energy, x.brief.mood, x.brief.mode, str(x.brief.bpm), x.brief.key_name)
           for b, x in load(ROOT / "drafts" / "v1" / "briefs.yaml").items()}
    assert {r[0]: r[1:] for r in rows} == got and len(rows) == 10


def test_a_card_per_persona():
    for p in PERSONAS:
        assert re.search(rf"^### 2\.\d Card \d: `{p}`", DOC, re.M), p


def test_the_vocabulary_and_the_chord_lists_are_the_palettes():
    v = P.vocabulary(P.load())
    for mood, chords in v["chords_in_mode"].items():
        m = re.search(rf"^- \*\*{mood}\*\* \(\w+\): (.+?)(?=^- \*\*|^\n|\Z)", DOC, re.M | re.S)
        assert m and m.group(1).split() == chords, mood
    for word in [*v["keys"], *v["kits"], *v["voicings"]]:
        assert f"`{word}`" in DOC
    for mood, ranks in P.VAMP_RANKS.items():
        for tie in ranks:
            for name, chords, span in tie:
                assert f"{name} `{chords} span {span}`" in DOC


def test_the_levels_table_is_the_palettes():
    dp = P.load()
    rows = dict(re.findall(r"^\| (?:keys|kit|sax) `(\w+)`[^|]*\| `(\w+)`[^|]*\|", DOC, re.M))
    assert rows == {**{k: v.option for k, v in dp.keys.items()}, **{k: v.option for k, v in dp.kits.items()},
                    "alto": dp.sax.option}
    for lv in [*dp.keys.values(), *dp.kits.values(), *dp.basses.values(), dp.sax]:
        assert f"{lv.fader_db:+.2f}".replace("-", "−") in DOC or f"{lv.fader_db:.3f}".replace("-", "−") in DOC, lv
