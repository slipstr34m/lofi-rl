"""The copy penalty P_copy of the GRPO reward (docs/design/taste_pipeline.md §5, Penalties; D-020, Before GRPO run 1).

    P_copy = 0                      for J <= 0.6
             0.3 * (J - 0.6) / 0.2  for 0.6 < J < 0.8
             0.3                    for J >= 0.8

J is the largest character 5-gram Jaccard of a completion with any reference, read by lofi_rl.rl.junk's measure (its
grams, jaccard, Reference and best_match, imported, never copied), so the reward sees copying exactly as the
base-model test counted it:
- **The completion's copy form** is junk.score_row's. A completion whose text passes the parser's gates (dsl.parse
  under the palette's vocabulary in the brief's mode; a completion cut at the token limit is G2, as finish_reason
  "length" is there) is compared by its canonical text without `sax:`, the form the examples are printed in. Any
  other is compared by its raw text without fences, blank lines or extra spaces (junk.plain_text).
- **The references** are every complete loop the run's prompt version can show, because across its prompts the
  policy sees all of them: v1 its one example (prompt.example_text), v2 all 9 of prompt_v2.reference_texts (strict:
  a palette that refuses one refuses the run). The optional pool (the seed pool, later) adds one reference per `.ls`
  file, in that file's own copy form under no brief's mode.
- **The nearest reference** is the one with the largest J, the first in reference order on a tie (best_match).

taste_pipeline §5 names MinHash; this is the exact Jaccard that MinHash estimates. Every reference is compared with
every completion, which is 9 comparisons for v2 and fine for a pool of a few thousand loops (an LSH index is the
change to make if the pool's cost shows in the step time).

Standard library and the project's standard-library modules only, so it imports on the laptop and in the policy env.
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lofi_rl.brief import Brief
from lofi_rl.dsl import canon, parse
from lofi_rl.palette import Palette
from lofi_rl.rl import junk

J_FREE, J_FULL, P_MAX = 0.6, 0.8, 0.3          # taste_pipeline §5: "0 at 0.6, linear to 0.3 at >=0.8"
NGRAM = junk.NGRAM                              # 5: character 5-grams, as junk's near-copy rule reads them
RULE = ("P_copy = 0 at J <= 0.6, rising linearly to 0.3 at J >= 0.8; J = the largest character 5-gram Jaccard "
        "(lofi_rl.rl.junk) of the completion's copy form with every example the prompt version can show and the pool")
VERSIONS = ("v1", "v2")


def ramp(j: float) -> float:
    """P_copy for a Jaccard j: 0 up to J_FREE, P_MAX from J_FULL, linear between. The endpoints come back exactly."""
    if j <= J_FREE:
        return 0.0
    if j >= J_FULL:
        return P_MAX
    return min(P_MAX, P_MAX * (j - J_FREE) / (J_FULL - J_FREE))


def copy_form(text: str, mode: str | None, palette: Palette, truncated: bool | None = None) -> str:
    """The text junk.score_row compares: the canonical text without `sax:` when the parser's gates pass (a cut
    completion is G2, as finish_reason "length" is in junk), else junk.plain_text of the raw text. A parser crash is
    a failed gate here, as it is in junk ("crash")."""
    try:
        res = parse(text, vocab=palette.vocab(), finish_reason="length" if truncated else None, mode=mode)
    except Exception:  # noqa: BLE001: junk counts a parser crash as a gate failure and compares the plain text
        return junk.plain_text(text)
    if res.gate.ok and res.ast is not None:
        return canon(dataclasses.replace(res.ast, sax=None))
    return junk.plain_text(text)


@dataclass(frozen=True)
class CopyResult:
    p_copy: float
    j: float                    # the largest Jaccard with any reference (0.0 with none)
    nearest: str | None         # the reference that gives it


class CopyPenalty:
    """P_copy against a fixed set of references (module docstring). Build it with for_prompt(); call it once per
    completion: copy(text, brief, truncated) -> CopyResult."""

    def __init__(self, references: Mapping[str, str], palette: Palette, *, version: str | None = None,
                 pool: Sequence[str] = ()) -> None:
        if not references:
            raise ValueError("the copy penalty needs at least one reference")
        self.palette, self.version, self.pool = palette, version, tuple(pool)
        self.texts = dict(references)
        self.references = tuple(junk.Reference.of(ref_id, text) for ref_id, text in self.texts.items())

    @classmethod
    def for_prompt(cls, version: str, palette: Palette, pool_files: Iterable[Path | str] = (),
                   root: Path | None = None, module: Any = None) -> CopyPenalty:
        """Every example prompt `version` can show (v1: its one; v2: all 9), read from `module` (the trainer's prompt
        module; by default lofi_rl.prompt or lofi_rl.prompt_v2), then one reference per pool file, keyed by its path
        (relative to `root` when it lies under it)."""
        refs = prompt_references(version, palette, module)
        pool = []
        for f in pool_files:
            p = Path(f)
            key = str(p.relative_to(root)) if root is not None and p.is_relative_to(root) else str(p)
            if key in refs:
                raise ValueError(f"the pool file {key} has the id of a prompt example")
            refs[key] = copy_form(p.read_text(encoding="utf-8"), None, palette)
            pool.append(key)
        return cls(refs, palette, version=version, pool=pool)

    def __call__(self, text: str, brief: Brief, truncated: bool | None = None) -> CopyResult:
        j, nearest = junk.best_match(junk.grams(copy_form(text, brief.mode, self.palette, truncated)), self.references)
        return CopyResult(ramp(j), j, nearest)

    def summary(self) -> dict[str, Any]:
        """What the dry run and run.json record: the rule, its numbers and each reference's id and text sha256."""
        prompt_ids = [k for k in self.texts if k not in self.pool]
        return {"rule": RULE, "j_free": J_FREE, "j_full": J_FULL, "p_max": P_MAX, "ngram": NGRAM,
                "measure": "lofi_rl.rl.junk (grams, jaccard, best_match)", "prompt_version": self.version,
                "prompt_examples": prompt_ids, "pool": {"files": len(self.pool), "ids": list(self.pool[:20])},
                "references": len(self.texts),
                "sha256": {k: hashlib.sha256(t.encode("utf-8")).hexdigest() for k, t in self.texts.items()
                           if k in prompt_ids}}


def prompt_references(version: str, palette: Palette, module: Any = None) -> dict[str, str]:
    """{id: text} of every complete loop prompt `version` can show, as the prompts print them (canonical, no sax):
    v1's one example (junk.shown_examples' id "example"), or all 9 of prompt_v2.reference_texts. `module` is the
    template module to read them from (default: lofi_rl.prompt for v1, lofi_rl.prompt_v2 for v2)."""
    if version == "v1":
        if module is None:
            from lofi_rl import prompt as module
        return {"example": module.example_text(palette)}
    if version == "v2":
        if module is None:
            from lofi_rl import prompt_v2 as module
        return dict(module.reference_texts(palette, strict=True))
    raise ValueError(f"prompt version {version!r} is not one of {VERSIONS}")
