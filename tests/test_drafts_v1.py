"""The 30 keep3 drafts as committed (drafts/v1; docs/DRAFTS.md §2.1 and §5), and each redraft folder beside them (§6:
drafts/v1/redo1, its briefs only, checked against drafts/v1). They pass the validator, and they keep the
house-rule readings that the validator reports (rubs) or that the text shows but nothing fails on: no sax note rubs
against the band, every answer ends on another pitch than its call, no stretch of more than 3 bars goes without a rest,
and bar 5, where the answer restates the call's first bar over the same chord, brings more than a ghost or a level.
Structure only, from the parsed and compiled loops: nothing here is a claim about how any draft sounds."""
from pathlib import Path

import pytest

from lofi_rl.compile.compiler import compile_loop
from lofi_rl.drafts import check as C
from lofi_rl.drafts.palette import load as load_palette
from lofi_rl.dsl import parse
from lofi_rl.dsl.vocab import LANES
from lofi_rl.verify.view import make_view

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "drafts" / "v1"
MAX_RUN = 48                    # 3 bars of 16 steps


# folder -> (briefs in scope, the folders it is checked against, drafts expected)
FOLDERS = {
    V1: (None, (), 30),
    V1 / "redo1": (["d01", "d06", "d07", "d10"], (V1,), 12),
}


@pytest.fixture(scope="module", params=list(FOLDERS), ids=lambda f: f.relative_to(ROOT).as_posix())
def folder(request):
    return request.param


@pytest.fixture(scope="module")
def report(folder):
    scope, against, _ = FOLDERS[folder]
    return C.check_folder(folder, scope=scope, against=list(against))


@pytest.fixture(scope="module")
def views(report):
    pal = load_palette().palette
    out = {}
    for d in report.drafts:
        res = parse(d.text, vocab=pal.vocab(), mode=d.brief.mode)
        out[d.file] = make_view(res.ast, d.brief, compile_loop(res.ast, d.brief, pal))
    return out


def test_every_draft_passes_the_validator(folder, report):
    assert [(d.file, d.problems) for d in report.drafts if not d.ok] == []
    assert report.ok and report.missing == [] and len(report.drafts) == FOLDERS[folder][2]


def test_no_sax_note_rubs_against_the_band(report):
    assert {d.file: d.info["rubs"] for d in report.drafts if d.info["rubs"]} == {}


def test_every_answer_ends_on_another_pitch_than_its_call(views):
    for f, v in views.items():
        a, b = ([i for i in v.phrase(p) if not i.rest][-1] for p in ("A", "B"))
        assert a.pc != b.pc, f


def test_no_stretch_of_more_than_three_bars_without_a_rest(views):
    for f, v in views.items():
        run = longest = 0
        for item in v.items * 2:                # the call runs into the answer, and the answer into the call
            run = 0 if item.rest else run + item.dur
            longest = max(longest, run)
        assert longest <= MAX_RUN, (f, longest)


def _hits(cells: str) -> set[int]:
    return {k for k, c in enumerate(cells) if c in "Xx"}


def test_bar_5_brings_more_than_a_ghost_or_a_level(views):
    for f, v in views.items():
        def sax(start):
            return [(i.step - start, i.dur, i.rest, i.pitch) for i in v.items if start <= i.step < start + 16]
        assert (v.bars.chords[4] != v.bars.chords[0] or sax(64) != sax(0) or v.comp[4] != v.comp[0]
                or v.bass[4] != v.bass[0] or any(_hits(v.drum(x, 4)) != _hits(v.drum(x, 0)) for x in LANES)), f
