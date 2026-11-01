"""The 10 draft briefs (docs/DRAFTS.md §1): city at night, every energy x mood cell once, plus one more.

    python -m lofi_rl.drafts.briefs [--seed 1105] [--out drafts/v1/briefs.yaml]    # writes the file
    python -m lofi_rl.drafts.briefs --check [drafts/v1/briefs.yaml]                # 0 when the file is the rule's

The rule (RULE), every draw a keyed SplitMix64 hash h(field, i) = splitmix64(splitmix64(splitmix64(seed) ^ field) ^ i)
(lofi_rl.prng), so one seed gives the same briefs on every machine:
1. **Cells.** d01-d09 are the 9 cells in ENERGIES x MOOD_NAMES order (low/melancholy, low/wistful, ..., high/hopeful);
   d10 is one more cell, CELLS[below(h(1, 0), 9)], each cell equally likely.
2. **Tempo.** A stratified draw of 10 from the brief sampler's own tempo weights (lofi_rl.brief.TEMPO_WEIGHTS: 60-95
   BPM, each tempo in 75-90 four times as likely as one outside it, TASTE.md Tempo/feel). The 84 weight units are cut
   into 10 strata [floor(84 k / 10), floor(84 (k + 1) / 10)); stratum k gives the tempo of unit
   lo + below(h(2, k), hi - lo). So the 10 tempos spread over the distribution as its tenths do (about 7.6 of 10 in
   75-90 on average, with no luck of the draw piling them up), then a keyed permutation (indices sorted by h(3, k))
   deals them to d01-d10.
3. **Tonic.** The 12 pitch classes sorted by (h(4, pc), pc), the first 10 dealt to d01-d10: every tonic is equally
   likely for each brief (TASTE.md, Claude's defaults: 12 tonics, uniform) and the 10 briefs sit in 10 different keys
   (TASTE.md Variation: different briefs should give different songs).

Mood picks the mode (melancholy aeolian, wistful dorian, hopeful ionian). The writers see only the scene and the two
knobs, as the policy does; tempo and tonic are the compiler's (instrument.md §1: never written by the model).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from lofi_rl.brief import ENERGIES, MOOD_NAMES, PITCH_NAMES, SCENE, TEMPO_WEIGHTS, Brief
from lofi_rl.palette import ROOT
from lofi_rl.prng import MASK, below, splitmix64

SEED = 1105                         # the keep3 render date, Nov 5 (docs/SHOOTOUTS.md §1 step 3)
COUNT = 10
CELLS: tuple[tuple[str, str], ...] = tuple((e, m) for e in ENERGIES for m in MOOD_NAMES)
FIELD_EXTRA, FIELD_TEMPO, FIELD_TEMPO_ORDER, FIELD_TONIC = 1, 2, 3, 4
BRIEFS_FILE = ROOT / "drafts" / "v1" / "briefs.yaml"
SCHEMA = "lofi-rl/draft-briefs/1"
RULE = ("d01-d09: the 9 energy x mood cells in ENERGIES x MOOD_NAMES order; d10: CELLS[below(h(1,0), 9)]. "
        "Tempo: a stratified draw of 10 from lofi_rl.brief.TEMPO_WEIGHTS (stratum k of the 84 weight units is "
        "[floor(84k/10), floor(84(k+1)/10)), its unit lo + below(h(2,k), hi - lo)), dealt to d01-d10 in the order "
        "of h(3,k). Tonic: the pitch classes sorted by (h(4,pc), pc), the first 10 dealt to d01-d10. "
        "h(field, i) = splitmix64(splitmix64(splitmix64(seed) ^ field) ^ i).")


@dataclass(frozen=True)
class DraftBrief:
    id: str                         # d01 .. d10
    brief: Brief

    @property
    def cell(self) -> tuple[str, str]:
        return self.brief.energy, self.brief.mood

    def as_dict(self) -> dict:
        b = self.brief
        return {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic, "key": b.key_name}


class BriefsError(ValueError):
    pass


def h(seed: int, field: int, i: int) -> int:
    return splitmix64(splitmix64(splitmix64(seed & MASK) ^ field) ^ i)


def tempo_of_unit(unit: int) -> int:
    """The tempo whose weight units hold `unit` (0-based, TEMPO_WEIGHTS order)."""
    for bpm, w in TEMPO_WEIGHTS:
        if unit < w:
            return bpm
        unit -= w
    raise ValueError("unit is past the tempo weights")


def stratified_tempos(seed: int, count: int = COUNT) -> list[int]:
    """One tempo per stratum of the tempo weights, in stratum order (slowest first)."""
    total = sum(w for _, w in TEMPO_WEIGHTS)
    out = []
    for k in range(count):
        lo, hi = total * k // count, total * (k + 1) // count
        out.append(tempo_of_unit(lo + below(h(seed, FIELD_TEMPO, k), hi - lo)))
    return out


def ids(count: int = COUNT) -> tuple[str, ...]:
    return tuple(f"d{i + 1:02d}" for i in range(count))


def draw(seed: int = SEED, count: int = COUNT) -> tuple[DraftBrief, ...]:
    """The briefs of the rule (RULE) for `seed`: the 9 cells, then the extra cells, each with its tempo and tonic."""
    if not len(CELLS) <= count <= 12:
        raise BriefsError(f"the rule covers the {len(CELLS)} cells once and deals distinct tonics: count must be "
                          f"{len(CELLS)}-12, got {count}")
    cells = list(CELLS) + [CELLS[below(h(seed, FIELD_EXTRA, j), len(CELLS))] for j in range(count - len(CELLS))]
    strata = stratified_tempos(seed, count)
    order = sorted(range(count), key=lambda k: (h(seed, FIELD_TEMPO_ORDER, k), k))
    tempos = [strata[k] for k in order]
    tonics = sorted(range(12), key=lambda pc: (h(seed, FIELD_TONIC, pc), pc))[:count]
    return tuple(DraftBrief(bid, Brief(energy=e, mood=m, bpm=t, tonic=pc))
                 for bid, (e, m), t, pc in zip(ids(count), cells, tempos, tonics))


def document(seed: int = SEED, count: int = COUNT) -> dict:
    return {"schema": SCHEMA, "seed": seed, "count": count, "scene": SCENE, "rule": RULE,
            "briefs": {b.id: b.as_dict() for b in draw(seed, count)}}


def text(seed: int = SEED, count: int = COUNT) -> str:
    head = ("# The keep3 draft briefs (docs/DRAFTS.md §1): written by python -m lofi_rl.drafts.briefs from the rule and\n"
            "# seed below; do not edit. The writers see only the scene and the two knobs (energy, mood), as the policy\n"
            "# does; bpm and tonic are the compiler's, and `key` is for logs only (lofi_rl.brief.Brief.key_name).\n")
    doc = document(seed, count)
    body = yaml.safe_dump({k: v for k, v in doc.items() if k != "briefs"}, sort_keys=False, width=120,
                          allow_unicode=True)
    rows = "".join(f"  {bid}: {{energy: {b['energy']}, mood: {b['mood']}, bpm: {b['bpm']}, tonic: {b['tonic']}, "
                   f"key: {b['key']}}}\n" for bid, b in doc["briefs"].items())
    return head + body + "briefs:\n" + rows


def load(path: Path | str = BRIEFS_FILE) -> dict[str, DraftBrief]:
    """The briefs of a briefs.yaml, by id, checked: the scene, each field a valid Brief, its key name its own."""
    p = Path(path)
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
        raise BriefsError(f"{p}: {err}") from err
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or not isinstance(doc.get("briefs"), dict):
        raise BriefsError(f"{p}: not a {SCHEMA} document with a `briefs` mapping")
    if doc.get("scene") != SCENE:
        raise BriefsError(f"{p}: scene {doc.get('scene')!r}; the only scene is {SCENE!r}")
    out = {}
    for bid, row in doc["briefs"].items():
        if not isinstance(row, dict) or set(row) != {"energy", "mood", "bpm", "tonic", "key"}:
            raise BriefsError(f"{p}: brief {bid}: expected energy, mood, bpm, tonic and key")
        try:
            b = Brief(energy=row["energy"], mood=row["mood"], bpm=row["bpm"], tonic=row["tonic"])
        except ValueError as err:
            raise BriefsError(f"{p}: brief {bid}: {err}") from err
        if row["key"] != b.key_name:
            raise BriefsError(f"{p}: brief {bid}: key {row['key']!r} is not {b.key_name!r}")
        out[str(bid)] = DraftBrief(str(bid), b)
    return out


def briefs_file_for(drafts_dir: Path | str) -> Path:
    """The briefs.yaml a drafts folder uses: its own, else its parent's (a redraft folder, docs/DRAFTS.md §6)."""
    d = Path(drafts_dir)
    for cand in (d / "briefs.yaml", d.parent / "briefs.yaml"):
        if cand.is_file():
            return cand
    raise BriefsError(f"{d}: no briefs.yaml in it or in its parent")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.drafts.briefs", description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--count", type=int, default=COUNT)
    p.add_argument("--out", type=Path, default=BRIEFS_FILE)
    p.add_argument("--check", nargs="?", const=str(BRIEFS_FILE), default=None, metavar="FILE",
                   help="exit 0 when FILE is exactly what the rule writes for its seed")
    a = p.parse_args(argv)
    try:
        if a.check is not None:
            path = Path(a.check)
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            want = text(int(doc["seed"]), int(doc["count"]))
            if path.read_text(encoding="utf-8") != want:
                print(f"{path}: not the rule's output for seed {doc['seed']}", file=sys.stderr)
                return 1
            load(path)
            print(f"{path}: ok, the rule's {doc['count']} briefs for seed {doc['seed']}")
            return 0
        out = text(a.seed, a.count)
    except (BriefsError, OSError, KeyError, TypeError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out, encoding="utf-8")
    print(f"wrote {a.out}")
    print(out, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
