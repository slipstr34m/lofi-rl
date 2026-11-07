"""The step-0 documents state the registered values (docs/design/step0_samples.md §6.1, test_step0_docs.py's row):
the counts, m per set, the rule name and the mark, the walltime and the cost, read from lofi_rl.step0 and its
requests, never typed in twice.

- **The build spec** (docs/design/step0_samples.md): the requests per stream under Q6(a) and (b), the sets' sizes, m
  per set in §1.2's table, the rule, the mark's parts, the job's shape, walltime and Cost line (CLAUDE.md §4's rates),
  §2.8's timeout sum, the steps' deadlines, the schemas and files of §2.6 and §4.4, and every field §4.1's validity
  table names, each read by a condition of lofi_rl.step0.check. Once jobs/step0.sbatch exists, its walltime and Cost
  line are the spec's (skipped before).
- **D-023** (docs/DECISIONS.md): the same values and the mark line itself, verbatim. D-023 is written with Shresth's
  answers to the spec's §7 (§6.2 step 4), so until then test_d023_states_the_registered_values is one of the
  branch's two expected failures (the other is test_step0_job's registry check).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import lofi_rl.step0 as S
from lofi_rl.step0 import check as C
from lofi_rl.step0 import cli
from lofi_rl.step0 import requests as R

ROOT = Path(__file__).resolve().parents[1]
SPEC = (ROOT / "docs" / "design" / "step0_samples.md").read_text(encoding="utf-8")
DECISIONS = ROOT / "docs" / "DECISIONS.md"
JOB = ROOT / S.JOB
Q6B = (("rm_heldout", "all"), ("train", "pool"))
COST = re.compile(r"Cost: (\d+) CPU \+ (\d+) \((\d+) GB\) \+ 8 \(1 GPU\) = (\d+) SU/min x (\d+) min -> at most "
                  r"(\d+) SU")                         # tests/test_junk_job.py's
# CLAUDE.md §4: GPU 8 SU/min each; CPU 1 SU/core-min; memory 1 SU per 4 GB-min. Claude registers a type up to 1,000 SU
GPU_SU, GB_PER_SU, CLAUDE_MAX = 8, 4, 1000


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def section(title: str) -> str:
    """The spec's text from a heading that starts with `title` to the next heading of its level or above."""
    m = re.search(rf"^(#+) {re.escape(title)}", SPEC, re.M)
    assert m, title
    rest = SPEC[m.end():]
    end = re.search(rf"^#{{1,{len(m.group(1))}}} ", rest, re.M)
    return rest[:end.start()] if end else rest


def counts() -> dict[str, int]:
    """The registered counts, from the code: the requests per stream (Q6(a) and (b)) and the sets' sizes."""
    reqs = R.requests()
    b = R.requests(drawn=Q6B, mark=R.mark_line(Q6B))
    held, train = (sum(1 for r in reqs if r.stream == s) for s in ("rm_heldout", "train"))
    briefs = R.stream_counts(reqs)
    return {"held": held, "train": train, "all": len(reqs), "train_b": sum(1 for r in b if r.stream == "train"),
            "all_b": len(b), "held_briefs": briefs["rm_heldout"]["briefs"], "train_briefs": briefs["train"]["briefs"],
            "heldout_clips": briefs["rm_heldout"]["briefs"] * S.M["heldout"],
            "refb_clips": briefs["train"]["briefs"] * S.M["refb"], "pool_clips": S.POOL_N * S.M["pool"]}


def cost_of(cpus: int, gb: int, minutes: int) -> tuple[int, int]:
    per_min = cpus + gb // GB_PER_SU + GPU_SU
    return per_min, per_min * minutes


# --------------------------------------------------------------------------------------------- the build spec
def test_the_spec_states_the_requests_and_the_sets_sizes():
    n = counts()
    assert (n["held"], n["train"], n["all"], n["train_b"], n["all_b"]) == (320, 3200, 3520, 1760, 2080)
    text = flat(SPEC)
    assert f"rm_heldout {n['held_briefs']} × {S.DRAWS} = {n['held']}" in text
    assert f"train {n['train_briefs']} × {S.DRAWS} = {n['train']:,}: {n['all']:,} in all under Q6(a)" in text
    assert f"{n['held']} + {n['train_b']:,} = {n['all_b']:,}" in text
    assert f"{n['heldout_clips']} clips if no brief is short" in text
    assert f"so {n['refb_clips']:,} in all" in text
    assert f"{S.POOL_N}×{S.M['pool']}" in text and f"seed pool's {n['pool_clips']}" in text
    assert f"(a) **Recommended.** Draw all {n['train_briefs']} training briefs now" in text
    assert f"Stopping at m would need at least {n['heldout_clips']} + {n['refb_clips']:,} + {n['pool_clips']} = " \
           f"{n['heldout_clips'] + n['refb_clips'] + n['pool_clips']:,} requests" in text


def test_the_spec_states_m_and_the_draws_of_each_set():
    table = section("1.2 The three sets").split("| set |", 1)[1].split("\n\n", 1)[0]
    rows = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 5 and cells[0] in ("RM held-out", "Ref_b", "seed pool's 660"):
            rows[cells[0]] = (int(re.search(r"\d+", cells[3]).group()), int(cells[4]))   # "the same 8 as Ref_b"
    assert rows == {"RM held-out": (S.DRAWS, S.M["heldout"]), "Ref_b": (S.DRAWS, S.M["refb"]),
                    "seed pool's 660": (S.DRAWS, S.M["pool"])}


def test_the_spec_states_the_rule_and_the_marks_parts():
    assert S.RULE == "step0-select/1" and f"rule `{S.RULE}`" in SPEC
    assert "`MARK`, the D-023 mark" in SPEC and "equal to `lofi_rl.step0.MARK`" in SPEC
    assert "one line naming the rule and the streams drawn" in flat(SPEC)
    assert S.MARK == R.mark_line(S.STREAMS, S.RULE) and S.MARK.startswith(f"D-023 mark: rule {S.RULE}; streams ")
    quoted = re.findall(r"D-023 mark: [^`\n]*", SPEC)
    assert all(q.rstrip(" .") == S.MARK for q in quoted), "a mark the spec quotes is MARK"


def test_the_spec_states_the_jobs_shape_walltime_and_cost():
    shape = flat(section("2.1 Shape and cost"))
    for line in ("`--job-name=lofi-step0`", "`--partition=gpu`", "`--cpus-per-task=4`", "`--mem=24G`",
                 "`--gres=gpu:a40:1`", "`--time=00:45:00`", "`--no-requeue`"):
        assert line in shape, line
    m = COST.search(shape)
    assert m, "the Cost line"
    cpus, mem_su, gb, per_min, minutes, most = map(int, m.groups())
    assert (cpus, gb, minutes) == (4, 24, 45) and mem_su == gb // GB_PER_SU
    assert cost_of(cpus, gb, minutes) == (per_min, most) == (18, 810) and most <= CLAUDE_MAX
    assert f"`make approve JOB={S.JOB} MAX={most} BY=Claude`" in shape
    job = next(x for x in SPEC.splitlines() if x.startswith("- **Job:**"))
    assert f"`{S.JOB}` (`{S.JOB_NAME}`): one A40, {minutes} min, at most {most} SU, registered by Claude" in job


def test_the_spec_states_the_deadlines_and_a_minute_spare():
    table = section("2.8 Deadlines, timeouts and the walltime")
    assert f"generate --deadline-s {S.GENERATE_DEADLINE_S:.0f}" in table
    assert f"render --deadline-s {S.RENDER_DEADLINE_S:.0f}" in table
    budgets = [int(x.replace(",", "")) for x in re.findall(r"\| ([\d,]+) s \|", table)]
    assert budgets == [1095, 430, 970, 94]
    total = sum(budgets)
    assert f"{total:,} s: {45 * 60 - total} s under {45 * 60:,} s (45 min)" in table and 45 * 60 - total >= 60


def test_the_spec_states_the_files_and_their_schemas():
    files = flat(section("2.6 Files and schemas"))
    for name in S.SUMMARY_FILES:
        assert f"`{name}`" in files, name
    for name, schema in S.SCHEMAS.items():
        assert f"| `{name}` | `{schema}` |" in files, name
    out = flat(section("4.4 Fixing the outputs"))
    assert f"`{S.HELDOUT_FILE}`** (`{S.HELDOUT_SCHEMA}`)" in out and f"`{S.RUN_FILE}`** (`{S.RUN_SCHEMA}`)" in out
    assert f"`{S.PATH_FILE}`" in flat(SPEC)


def test_the_specs_commands_are_the_command_lines():
    usage = {"check": "check RUN_DIR [--redo REDO_DIR] [--json]", "prereg": "prereg runs/step0-<job> [--redo"}
    for cmd, text in usage.items():
        assert text in SPEC, cmd
    p = cli.parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    assert {"--redo", "--json", "--rescore"} <= set(sub.choices["check"]._option_string_actions)
    assert {"--redo", "--write"} <= set(sub.choices["prereg"]._option_string_actions)
    assert "--force" not in sub.choices["prereg"]._option_string_actions, "§4.4: the writer has no --force"
    assert "`check --rescore`" in SPEC and "STEP0_CHECK <verdict>" in SPEC


# the fields §4.1's validity table names (as it writes them) -> the conditions of lofi_rl.step0.check that read them
FIELDS = {
    "generate.stopped": ["generate.stopped"], "generate.partial": ["generate.partial"], "n": ["n"],
    "requested": ["requested", "generate.requested"], "unreadable_lines": ["unreadable_lines"],
    "requests.missing": ["requests.missing"], "requests.duplicate": ["requests.duplicate"],
    "requests.foreign": ["requests.foreign"], "requests.seed_mismatch": ["requests.seed_mismatch"],
    "probe.passed": ["generate.probe.passed"], "prompt_tokens": ["requests.prompt_tokens_mismatch"],
    "generate.model.revision": ["generate.model.revision"], "streams_file.sha256": ["generate.streams_file.sha256"],
    "fresh": ["generate.streams_file.fresh"], "sample_brief(seed, index)": ["requests.brief_mismatch"],
    "prompt.version": ["by_stream.{s}.prompt.version"], "template_sha256": ["by_stream.{s}.prompt.template_sha256"],
    "template_matches_generate": ["by_stream.{s}.prompt.template_matches_generate"],
    "rows_matching": ["by_stream.{s}.prompt.rows_matching"],
    "example_mismatch": ["by_stream.{s}.prompt.example_mismatch"],
    "palette.sha256": ["generate.palette.sha256", "by_stream.{s}.palette.sha256", "clips.palette.sha256"],
    "matches_generate": ["by_stream.{s}.palette.matches_generate", "by_stream.{s}.constrained.in_language."
                         "matches_generate"],
    "sampling": ["generate.sampling"], "registration.found": ["generate.registration.found"],
    "step0.MARK": ["generate.registration.mark", "docs/DECISIONS.md at the generate commit"],
    "constrained.enabled": ["by_stream.{s}.constrained.enabled"],
    "regex_sha256": ["by_stream.{s}.constrained.regex_sha256"],
    "in_language.regex_sha256": ["by_stream.{s}.constrained.in_language.regex_sha256"],
    "in_language.matches_generate": ["by_stream.{s}.constrained.in_language.matches_generate"],
    "stop.out_of_language": ["by_stream.{s}.constrained.in_language.stop.out_of_language"],
    "other.n": ["by_stream.{s}.constrained.in_language.other.n"],
    "failures.G1": ["by_stream.{s}.failures.G1"], "G3": ["by_stream.{s}.failures.G3"],
    "G4": ["by_stream.{s}.failures.G4"], "G5": ["by_stream.{s}.failures.G5"], "crash": ["by_stream.{s}.failures.crash"],
    "commit=… dirty=0": [C.STAMP_NAME], "path_file.blobs_ok": ["generate.path_file.blobs_ok"],
    "docs/prereg/D-023_step0_path.txt": ["the path at the generate commit"],
    "render.errors_final": ["render.errors_final"], "render.skipped": ["render.skipped"],
    "render.other_gates": ["render.other_gates"], "rm_heldout": ["sets.heldout streams"],
    "train": ["sets.refb streams", "sets.pool streams"], "heldout_text": ["sets.pool held-out"],
    "step0.select": ["sets.json = step0.select"], "taste_pipeline.md:76": ["clip ids"],
    "render_code.source_sha8": ["render_code.source_sha8"], "features.source_sha8": ["render_code.source_sha8"],
}


def test_every_field_the_validity_table_names_is_a_condition_of_check(tmp_path):
    table = section("4.1 Validity").split("| group | conditions |", 1)[1].split("\n\n", 1)[0]
    named = set(re.findall(r"`([^`]+)`", table))
    for sha in (S.MODEL_REVISION, S.STREAMS_SHA256, S.TEMPLATE_SHA256, S.PALETTE_SHA256, S.REGEX_SHA256):
        assert f"`{sha[:8]}…`" in table, sha
    assert set(FIELDS) <= named, sorted(set(FIELDS) - named)
    left = named - set(FIELDS) - {f"{sha[:8]}…" for sha in (S.MODEL_REVISION, S.STREAMS_SHA256, S.TEMPLATE_SHA256,
                                                           S.PALETTE_SHA256, S.REGEX_SHA256)}
    assert left == {"brief_streams.draw_seed(seed, index, k)", "draw_seed", "sp_seed", "DECISIONS.md:221"}, left
    import helpers_step0 as H
    run = C.load_run(H.make_run(tmp_path / "step0-1", briefs=["5101:0"]).dir)
    have = {c.name for c in C.conditions(run)[0]}
    for field, names in FIELDS.items():
        for name in names:
            for s in ("rm_heldout", "train"):
                assert name.format(s=s) in have, (field, name)
    assert len(have) == 74 and "the path at the redo commit" not in have


# --------------------------------------------------------------------------------------------- the job, once it exists
def test_the_job_script_has_the_walltime_and_cost_the_spec_states():
    if not JOB.is_file():
        pytest.skip(f"{S.JOB} is the cluster builder's: it is not in this tree yet")
    job = JOB.read_text(encoding="utf-8")
    spec = COST.search(flat(section("2.1 Shape and cost")))
    m = COST.search(job)
    assert m and m.groups() == spec.groups(), "the job's Cost line is the spec's"
    assert re.search(r"^#SBATCH --time=00:45:00$", job, re.M) and f"--job-name={S.JOB_NAME}" in job


# --------------------------------------------------------------------------------------------- D-023
def d023_entry(text: str) -> str | None:
    """D-023's entry in a DECISIONS.md text, up to the next entry's heading (whitespace folded), or None."""
    m = re.search(r"^## D-023 \(", text, re.M)
    if m is None:
        return None
    entry = text[m.start():]
    nxt = re.search(r"^## D-\d+ \(", entry[3:], re.M)
    return flat(entry if nxt is None else entry[:nxt.start() + 3])


def assert_d023_states_the_registered_values(text: str) -> None:
    entry = d023_entry(text)
    assert entry is not None, ("docs/DECISIONS.md has no D-023 yet: it is written with Shresth's answers to the "
                               "spec's section 7, beside the code (spec section 6.2 step 4)")
    n = counts()
    assert S.MARK in entry, "the mark, verbatim"
    assert f"`{S.RULE}`" in entry
    for key in ("all", "held", "train", "heldout_clips", "refb_clips", "pool_clips"):
        assert f"{n[key]:,}" in entry, key
    for name, m_ in S.M.items():
        assert f"m = {m_}" in entry or f"m {m_}" in entry, name
    assert f"`{S.JOB_NAME}`" in entry and "45 min" in entry and "at most 810 SU" in entry
    for rel in (S.PATH_FILE, S.HELDOUT_FILE, S.RUN_FILE):
        assert rel in entry, rel


def test_d023_states_the_registered_values():
    """D-023 holds the mark verbatim, the rule, the counts, m per set, the job, its walltime and its cost. It is
    written with Shresth's answers (spec §6.2 step 4): until then this test fails, as the branch expects."""
    assert_d023_states_the_registered_values(DECISIONS.read_text(encoding="utf-8"))
