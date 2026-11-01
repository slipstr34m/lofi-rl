"""The draft validator (docs/DRAFTS.md §5).

    python -m lofi_rl.drafts.check <dir> [--briefs d04+d07] [--against <dir> ...] [--palette palettes/draft_day2.yaml]
        [--json <out.json>]

<dir> holds drafts named `<brief id>_<persona id>.ls` (lofi_rl.drafts.personas) and a briefs.yaml (or its parent
does: a redraft folder, §6). Every draft of the briefs in scope (all of them, or `--briefs`) is checked, in order:
1. **name**: `<brief id>_<persona id>.ls`, the brief in the briefs file and the persona one of the three;
2. **gate**: dsl.parse with the draft palette's vocabulary in the brief's mood passes G1-G6 with no repair at all
   (a draft is written in the language, not repaired into it: no free alias, no padded grid), and has no `sax:` line
   (it chooses nothing, DSL.md §4, and prompt v1 shows none);
3. **compile**: compile_loop against the draft palette and the brief, with no repair (no sax octave fold);
4. **energy**: E's band is the brief's energy (verify.knobs, the provisional bands 0.38 and 0.59);
5. **mood**: the brief's mode wins the mood signature by 2 or more (verify.knobs);
6. **refs**: character 5-gram Jaccard below 0.6 against every prompt example (prompt v1's and the 9 of
   prompt_v2.reference_texts, in the draft palette's names) and every shootout reference loop (shootouts/loops);
7. **drafts**: the same below 0.6 against every other draft of <dir> and of each `--against` folder;
and, over the scope, **one draft per brief per persona**: exactly one file for each (brief, persona).

The Jaccard is junk-rate's near-copy measure (lofi_rl.rl.junk.grams and jaccard: character 5-grams) on each loop's copy
form: its canonical text with no `sax:` line and with the keys instrument and the kit written `K` and `D`, so naming
another instrument never lowers it. A file that does not parse is compared as plain text (junk.plain_text).

E, the mood margin, theory and P_sax are printed too; they are information, never failures. The table goes to stdout,
one line per problem after it; `--json` writes everything. Exit 0 when every check passes, 1 when one fails, 2 for bad
arguments (no folder, no briefs file, a palette that does not load). Sound claims: none.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path

import yaml

from lofi_rl import prompt, prompt_v2
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import CompileError, compile_loop
from lofi_rl.dsl import DEFAULT_VOCAB, Loop, canon, parse
from lofi_rl.dsl.vocab import MOODS
from lofi_rl.palette import ROOT, Palette
from lofi_rl.rl.junk import grams, jaccard, plain_text
from lofi_rl.verify.knobs import ENERGY_BANDS, energy_band
from lofi_rl.verify.scoring import score

from . import briefs as B
from .palette import DRAFT_PALETTE, DraftPaletteError, load
from .personas import DRAFT_RE, PERSONAS, draft_name

SCHEMA = "lofi-rl/draft-check/1"
J5_MAX = 0.6                                    # a draft's Jaccard must stay below this
SHOOTOUT_LOOPS = ROOT / "shootouts" / "loops"
CHECKS = ("name", "gate", "compile", "energy", "mood", "refs", "drafts")


def copy_form(loop: Loop) -> str:
    """Canonical text with no `sax:` line and the keys instrument and kit written K and D."""
    return canon(replace(loop, sax=None, keys=replace(loop.keys, instrument="K"), kit="D"))


@dataclass
class Draft:
    file: str                                   # the file name in its folder
    path: Path
    text: str
    brief_id: str | None = None
    persona: str | None = None
    brief: Brief | None = None
    loop: Loop | None = None
    form: str = ""                              # the copy form, or plain text when it does not parse
    problems: dict[str, str] = field(default_factory=dict)      # check -> what is wrong
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class Reference:
    id: str
    form: str
    grams: frozenset


def references(palette: Palette, loops_dir: Path = SHOOTOUT_LOOPS) -> list[Reference]:
    """Every prompt example (v1's, then v2's nine in their order) and every shootout reference loop, as copy forms.
    v1's example is v2's `example` too; it is listed once."""
    out, seen = [], set()

    def add(rid: str, text: str, mood: str, vocab) -> None:
        res = parse(text, vocab=vocab, mode=MOODS[mood])
        if not res.gate.ok:
            raise ValueError(f"reference {rid} does not pass the gates: {res.gate.code} {res.gate.detail}")
        form = copy_form(res.ast)
        if form not in seen:
            seen.add(form)
            out.append(Reference(rid, form, grams(form)))

    v = palette.vocab()
    add("prompt_v1:example", prompt.example_text(palette), prompt.EXAMPLE_BRIEF["mood"], v)
    cells = {ex_id: cell for cell, (ex_id, _) in prompt_v2.EXAMPLES.items()}
    for ex_id, text in prompt_v2.reference_texts(palette, strict=True).items():
        add(f"prompt_v2:{ex_id}", text, cells[ex_id][1], v)
    loops = yaml.safe_load((loops_dir / "briefs.yaml").read_text(encoding="utf-8"))["loops"]
    for name, b in sorted(loops.items()):
        add(f"shootout:{name}", (loops_dir / f"{name}.ls").read_text(encoding="utf-8"), b["mood"], DEFAULT_VOCAB)
    return out


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_drafts(folder: Path, briefs: dict[str, B.DraftBrief]) -> list[Draft]:
    """Every *.ls file of `folder`, named and parsed (its brief's mood); nothing else is checked here."""
    out = []
    for path in sorted(folder.glob("*.ls")):
        d = Draft(path.name, path, _read(path))
        m = DRAFT_RE.fullmatch(path.name)
        if not m:
            d.problems["name"] = "not <brief id>_<persona id>.ls"
        else:
            d.brief_id, d.persona = m["brief"], m["persona"]
            if d.brief_id not in briefs:
                d.problems["name"] = f"brief {d.brief_id} is not in the briefs file"
            elif d.persona not in PERSONAS:
                d.problems["name"] = f"persona {d.persona} is not one of {list(PERSONAS)}"
            else:
                d.brief = briefs[d.brief_id].brief
        out.append(d)
    return out


def check_one(d: Draft, palette: Palette) -> None:
    """Checks 2-5 on one named draft, filling d.problems, d.info, d.loop and d.form."""
    d.form = plain_text(d.text)
    if d.brief is None:
        return
    b = d.brief
    res = parse(d.text, vocab=palette.vocab(), mode=b.mode)
    d.info["repairs"] = sorted({r.code for r in res.repairs})
    if not res.gate.ok:
        d.problems["gate"] = f"{res.gate.code}: {res.gate.detail}"
        return
    d.loop = res.ast
    d.form = copy_form(res.ast)
    gate = []
    if res.repairs:
        gate.append("repaired (" + ", ".join(f"{r.code} line {r.line}" for r in res.repairs) +
                    "): write it in the language")
    if res.ast.sax is not None:
        gate.append("has a sax: line; leave it out (the palette's one sax plays every loop)")
    if gate:
        d.problems["gate"] = "; ".join(gate)
    try:
        compiled = compile_loop(res.ast, b, palette)
    except CompileError as err:
        d.problems["compile"] = str(err)
        return
    if compiled.repairs:
        d.problems["compile"] = "the sax line needed " + ", ".join(
            f"{r.code} {r.before}->{r.after}" for r in compiled.repairs) + " to fit the alto's range"
    d.info["digest"] = compiled.digest()
    s = score(d.text, b, palette)
    k = s.knob_parts
    band = energy_band(k["E"])
    d.info.update(E=round(k["E"], 4), band=band, mood_margin=k["mood_margin"], theory=round(s.theory, 3),
                  p_sax=round(s.p_sax, 3), theory_parts={x: round(v, 3) for x, v in s.theory_parts.items()})
    if band != b.energy:
        d.problems["energy"] = (f"E {k['E']:.3f} is {band}; the brief is {b.energy} (bands {ENERGY_BANDS[0]} / "
                                f"{ENERGY_BANDS[1]})")
    if k["f_mood"] != 1.0:
        d.problems["mood"] = (f"the {b.mode} mood signature wins by {k['mood_margin']:g}; it needs 2 (aeolian "
                              f"{k['mood_aeolian']:g}, dorian {k['mood_dorian']:g}, ionian {k['mood_ionian']:g})")


@dataclass
class Report:
    folder: Path
    scope: list[str]
    drafts: list[Draft]
    missing: list[str]                          # "<brief>_<persona>.ls" with no file
    against: list[Draft]
    n_refs: int = 0                             # references compared with
    n_pool: int = 0                             # drafts compared with each other (every file of the folders)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.errors and all(d.ok for d in self.drafts)

    def as_dict(self) -> dict:
        return {"schema": SCHEMA, "folder": str(self.folder), "scope": self.scope, "ok": self.ok,
                "j5_max": J5_MAX, "references": self.n_refs, "pool": self.n_pool, "missing": self.missing,
                "errors": self.errors,
                "drafts": [{"file": d.file, "brief": d.brief_id, "persona": d.persona, "ok": d.ok,
                            "problems": d.problems, **d.info} for d in self.drafts],
                "against": [d.file for d in self.against]}


def check_folder(folder: Path | str, *, scope: list[str] | None = None, against: list[Path] = (),
                 palette_path: Path | str = DRAFT_PALETTE, loops_dir: Path = SHOOTOUT_LOOPS) -> Report:
    """Every check over a drafts folder. Raises BriefsError or DraftPaletteError for bad inputs."""
    folder = Path(folder)
    if not folder.is_dir():
        raise B.BriefsError(f"{folder} is not a folder")
    briefs = B.load(B.briefs_file_for(folder))
    scope = list(scope) if scope else list(briefs)
    unknown = [s for s in scope if s not in briefs]
    if unknown:
        raise B.BriefsError(f"--briefs {unknown} are not in the briefs file")
    dp = load(palette_path)
    pal = dp.palette
    every = read_drafts(folder, briefs)
    for d in every:
        check_one(d, pal)
    drafts = [d for d in every if d.brief_id in scope or "name" in d.problems]
    others: list[Draft] = []
    for extra in against:
        extra = Path(extra)
        if not extra.is_dir():
            raise B.BriefsError(f"--against {extra} is not a folder")
        if extra.resolve() == folder.resolve():
            raise B.BriefsError(f"--against {extra} is the drafts folder itself (each draft would match itself)")
        try:
            eb = B.load(B.briefs_file_for(extra))
        except B.BriefsError:
            eb = {}
        for d in read_drafts(extra, eb):
            check_one(d, pal)
            d.file = f"{extra.name}/{d.file}"
            others.append(d)
    refs = references(pal, loops_dir)
    for d in drafts:
        g = grams(d.form)
        j, rid = max(((jaccard(g, r.grams), r.id) for r in refs), default=(0.0, None))
        d.info.update(refs_j5=round(j, 4), refs_nearest=rid)
        if j >= J5_MAX:
            d.problems["refs"] = f"5-gram Jaccard {j:.3f} with {rid} (it must stay below {J5_MAX})"
    pool = [x for x in every if x.form] + [x for x in others if x.form]
    pg = {id(x): grams(x.form) for x in pool}
    best: dict[int, tuple[float, str | None]] = {id(d): (0.0, None) for d in drafts}
    for x, y in combinations(pool, 2):
        j = jaccard(pg[id(x)], pg[id(y)])
        for a, b in ((x, y), (y, x)):
            if id(a) in best and j > best[id(a)][0]:
                best[id(a)] = (j, b.file)
    for d in drafts:
        j, other = best[id(d)]
        d.info.update(drafts_j5=round(j, 4), drafts_nearest=other)
        if j >= J5_MAX:
            d.problems["drafts"] = f"5-gram Jaccard {j:.3f} with {other} (it must stay below {J5_MAX})"
    have = {(d.brief_id, d.persona) for d in every if "name" not in d.problems}
    missing = [draft_name(b, p) for b in scope for p in PERSONAS if (b, p) not in have]
    return Report(folder, scope, drafts, missing, others, n_refs=len(refs), n_pool=len(pool))


def table(rep: Report) -> str:
    cols = ("draft", "gate", "E", "band", "mood", "refs J5", "nearest ref", "drafts J5", "nearest draft", "theory",
            "P_sax", "verdict")
    rows = []
    for d in rep.drafts:
        i = d.info
        gate = "ok" if not ({"name", "gate", "compile"} & set(d.problems)) else \
            next(c for c in ("name", "gate", "compile") if c in d.problems)
        rows.append((d.file, gate, f"{i['E']:.3f}" if "E" in i else "-", i.get("band", "-"),
                     f"{i['mood_margin']:+g}" if "mood_margin" in i else "-",
                     f"{i['refs_j5']:.3f}" if "refs_j5" in i else "-", str(i.get("refs_nearest") or "-"),
                     f"{i['drafts_j5']:.3f}" if "drafts_j5" in i else "-", str(i.get("drafts_nearest") or "-"),
                     f"{i['theory']:.2f}" if "theory" in i else "-", f"{i['p_sax']:.2f}" if "p_sax" in i else "-",
                     "ok" if d.ok else "FAIL " + "+".join(c for c in CHECKS if c in d.problems)))
    rows += [(m, "missing", *["-"] * 9, "FAIL missing") for m in rep.missing]
    widths = [max(len(str(r[k])) for r in [cols, *rows]) for k in range(len(cols))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in [cols, *rows]]
    n_ok = sum(d.ok for d in rep.drafts)
    lines.append(f"{n_ok} of {len(rep.drafts)} drafts pass; {len(rep.missing)} missing; scope {'+'.join(rep.scope)}; "
                 f"5-gram Jaccard must stay below {J5_MAX} against {rep.n_refs} references and {rep.n_pool - 1} other "
                 f"drafts ({len(rep.against)} of them from --against)")
    return "\n".join(lines)


def parse_scope(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [x for x in text.replace(",", "+").split("+") if x]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.drafts.check", description=__doc__.split("\n\n")[0])
    p.add_argument("dir", type=Path, help="the drafts folder")
    p.add_argument("--briefs", default=None, help="the briefs in scope, joined by + (default: all)")
    p.add_argument("--against", type=Path, action="append", default=[], help="more drafts to compare with")
    p.add_argument("--palette", type=Path, default=DRAFT_PALETTE)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    try:
        rep = check_folder(a.dir, scope=parse_scope(a.briefs), against=a.against, palette_path=a.palette)
    except (B.BriefsError, DraftPaletteError, ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(table(rep))
    for d in rep.drafts:
        for c in CHECKS:
            if c in d.problems:
                print(f"{d.file}: {c}: {d.problems[c]}")
    for m in rep.missing:
        print(f"{m}: missing (one draft per brief per persona)")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep.as_dict(), indent=1) + "\n", encoding="utf-8")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
