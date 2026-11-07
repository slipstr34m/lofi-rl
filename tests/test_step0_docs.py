"""The step-0 documents state the registered values (docs/design/step0_samples.md §6.1, test_step0_docs.py's row):
the counts, m per set, the rule name and the mark, the walltime and the cost, read from lofi_rl.step0 and its
requests, never typed in twice.

- **The build spec** (docs/design/step0_samples.md): the requests per stream under Q6(a) and (b), the sets' sizes, m
  per set in §1.2's table, the rule, the mark's parts, the job's shape, walltime and Cost line (CLAUDE.md §4's rates),
  §2.8's timeout sum, the steps' deadlines, the schemas and files of §2.6 and §4.4, and every field §4.1's validity
  table names, each read by a condition of lofi_rl.step0.check. Once jobs/step0.sbatch exists, its walltime and Cost
  line are the spec's (skipped before).
- **D-023** (docs/DECISIONS.md): the same values, the mark line and the answers line verbatim and each read as
  generate reads the mark (requests.find_marks), m in each set's own bullet, check's condition counts, the amendment's
  form as check reads it, the separation rules and the RM loader's ask, and no draft placeholder or comment; a
  stand-in entry in the draft's form passes, and each form generate or the record would not take fails. D-023 is
  written with Shresth's answers to the spec's §7 (§6.2 step 4), so until then test_d023_states_the_registered_values
  is one of the branch's two expected failures (the other is test_step0_job's registry check).
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
    shape = flat(section("2.1 Shape and cost"))
    spec = COST.search(shape)
    m = COST.search(job)
    assert m and m.groups() == spec.groups(), "the job's Cost line is the spec's"
    assert re.search(r"^#SBATCH --time=00:45:00$", job, re.M) and f"--job-name={S.JOB_NAME}" in job
    for key, value in re.findall(r"^#SBATCH --([a-z-]+)(?:=(\S+))?$", job, re.M):     # finding R3-8: read, not typed
        if key in ("nodes", "ntasks"):
            assert f"{value} {key[:-1].replace('ntask', 'task')}" in shape, key
        else:
            assert (f"`--{key}={value}`" if value else f"`--{key}`") in shape, key
    code = "\n".join(x for x in job.splitlines() if x.strip() and not x.lstrip().startswith("#"))
    table = section("2.8 Deadlines, timeouts and the walltime")
    pairs = sorted(re.findall(r"timeout -k (\d+) (\d+)", code))
    assert pairs == sorted(re.findall(r"timeout -k (\d+) (\d+)", table)), "the job's timeouts are §2.8's"


# --------------------------------------------------------------------------------------------- D-023
def d023_raw(text: str) -> str | None:
    """D-023's entry in a DECISIONS.md text, up to the next entry's heading, as written (lines kept), or None."""
    m = re.search(r"^## D-023 \(", text, re.M)
    if m is None:
        return None
    entry = text[m.start():]
    nxt = re.search(r"^## D-\d+ \(", entry[3:], re.M)
    return entry if nxt is None else entry[:nxt.start() + 3]


def d023_entry(text: str) -> str | None:
    """D-023's entry with its whitespace folded, or None."""
    raw = d023_raw(text)
    return None if raw is None else flat(raw)


def bullet(raw: str, title: str) -> str:
    """The text of D-023's bullet whose bold head starts with `title`, its wrapped lines folded in ("" if none)."""
    lines = raw.splitlines()
    for j, line in enumerate(lines):
        body = line.lstrip()
        if body.startswith(f"- **{title}"):
            indent, out = len(line) - len(body), [body]
            for nxt in lines[j + 1:]:
                if not nxt.strip() or nxt.lstrip().startswith("- ") or len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                out.append(nxt.strip())
            return " ".join(out)
    return ""


def n_conditions() -> tuple[int, int]:
    """How many conditions check holds a run to, and a redo pair (one more: `the path at the redo commit`)."""
    import dataclasses
    import tempfile
    import helpers_step0 as H
    with tempfile.TemporaryDirectory(prefix="step0-docs-") as tmp:
        run = C.load_run(H.make_run(Path(tmp) / "step0-1", briefs=["5101:0"]).dir)
        one = C.conditions(run)[0]
        pair = C.conditions(dataclasses.replace(run, redo=Path(tmp) / "step0-redo-2"))[0]
    names = {c.name for c in pair}
    assert C.REDO_STAMP_NAME in names and "the path at the redo commit" in names and C.STAMP_NAME not in names
    return len(one), len(pair)


TEMPLATE = "D-023 redo amendment: run step0-<job>; changed <path>, <path>"


def assert_d023_states_the_registered_values(text: str) -> None:
    raw = d023_raw(text)
    assert raw is not None, ("docs/DECISIONS.md has no D-023 yet: it is written with Shresth's answers to the "
                             "spec's section 7, beside the code (spec section 6.2 step 4)")
    entry, n = flat(raw), counts()
    assert "[[PLACEHOLDER" not in raw and "<!--" not in raw, "the draft's placeholders and comment are gone"
    assert S.MARK in R.find_marks(raw), "the mark, on one line, as generate's pre-check reads it (finding R2-2)"
    assert R.answers_line() in R.find_marks(raw, R.ANSWERS_HEAD), "the answers line, from the constants (R1-3)"
    assert f"`{S.RULE}`" in entry
    for key in ("all", "held", "train", "heldout_clips", "refb_clips", "pool_clips"):
        assert f"{n[key]:,}" in entry, key
    for title, name in (("The RM held-out tuples", "heldout"), ("PAR's Ref_b", "refb"),
                        ("The seed pool's 660", "pool")):
        assert f"m = {S.M[name]}" in bullet(raw, title), (title, "m in its own set's bullet (finding R3-8)")
    one, pair = n_conditions()
    assert f"{one} conditions" in entry and f"a redo pair has {pair}" in entry, "check's count (finding R3-8)"
    assert TEMPLATE in raw, "the amendment's form, verbatim"
    filled = TEMPLATE.replace("<job>", "13800001").replace("<path>, <path>", f"{S.JOB}, src/lofi_rl/step0/select.py")
    assert C.amendment_files(filled, "step0-13800001") == sorted([S.JOB, "src/lofi_rl/step0/select.py"])
    assert "held-out data" in entry and "taken or not" in entry, "the separation rules (finding R3-2)"
    assert "`short`" in bullet(raw, "The RM loader"), "a short brief's clips are held-out clips too (finding R3-4)"
    assert f"`{S.JOB_NAME}`" in entry and "45 min" in entry and "at most 810 SU" in entry
    for rel in (S.PATH_FILE, S.HELDOUT_FILE, S.RUN_FILE):
        assert rel in entry, rel


def standin_d023() -> str:
    """A D-023 entry holding what assert_d023_states_the_registered_values asks, in the draft's words and form (the
    mark and the answers line alone in fenced blocks), so that the check is seen to be satisfiable."""
    n, (one, pair) = counts(), n_conditions()
    return f"""
## D-023 (a stand-in for tests/test_step0_docs.py): the step-0 samples, drawn by `{S.JOB_NAME}` under rule `{S.RULE}`

- **His answers** (the constants that carry them):

  ```
  {R.answers_line()}
  ```

- **Rule `{S.RULE}`:**
  - **The draws.** rm_heldout 40 × 8 = {n['held']:,} and train 400 × 8 = {n['train']:,}, so {n['all']:,} requests.
  - **The RM held-out tuples** (rm_heldout, seed 5101): m = {S.M['heldout']}. The set is {n['heldout_clips']} clips,
    40 tuples of 4, if no brief is short.
  - **PAR's Ref_b** (train, seed 596): m = {S.M['refb']}, so {n['refb_clips']:,} in all.
  - **The seed pool's 660** (train, {S.POOL_N} × m = {S.M['pool']}): {n['pool_clips']} is a planning count.
- **The mark:**

  ```
  {S.MARK}
  ```

- **The job:** `{S.JOB_NAME}`, one A40, 45 min, so at most 810 SU.
- **The path:** `{S.PATH_FILE}`.
- **Validity:** That is {one} conditions under either Q6 answer; a redo pair has {pair}.
- **The fix comes first:** one line per redo, verbatim:

  ```
  {TEMPLATE}
  ```

- **Separation:** every rm_heldout row, taken or not, is held-out data.
- **Fixing the outputs:** `{S.HELDOUT_FILE}` and `{S.RUN_FILE}`.
- **Asked of later units:**
  - **The RM loader:** every label on a clip heldout_v1.json lists, in `tuples` or in `short`, is dropped.
"""


def wrapped(text: str, at: int = 118) -> str:
    """The mark wrapped across two lines at about `at` columns, as DECISIONS.md's entries are wrapped."""
    cut = S.MARK.rfind(" ", 0, at)
    return text.replace(S.MARK, S.MARK[:cut] + "\n  " + S.MARK[cut + 1:])


REFUSED = [
    ("the mark wrapped across two lines", wrapped),
    ("text after the mark on its line", lambda t: t.replace(S.MARK, S.MARK + " (Q6(a))")),
    ("the mark in a sentence, unquoted", lambda t: t.replace(f"  ```\n  {S.MARK}\n  ```",
                                                             f"  The mark is {S.MARK}; generate checks it.")),
    ("Ref_b's m given as 3", lambda t: t.replace(f"m = {S.M['refb']}, so", "m = 3, so")),
    ("an answer the constants do not carry", lambda t: t.replace("Q3 PASSING hard;", "Q3 PASSING hard_not_copy;")),
    ("a placeholder left in", lambda t: t + "\n- **Outcome:** [[PLACEHOLDER: after the run]]\n"),
    ("the draft's comment left in", lambda t: t.replace("- **His answers**", "<!-- DRAFT -->\n- **His answers**")),
    ("another count of conditions", lambda t: t.replace("a redo pair has", "a redo pair has 7")),
    ("no separation rules", lambda t: t.replace("held-out data", "held-out")),
]


def test_the_d023_check_takes_the_drafts_form():
    assert_d023_states_the_registered_values(DECISIONS.read_text(encoding="utf-8") + standin_d023())


@pytest.mark.parametrize("why, edit", REFUSED, ids=[r[0] for r in REFUSED])
def test_the_d023_check_refuses_what_generate_or_the_record_would_not_take(why, edit):
    """Findings R2-2 and R3-8: the forms of the mark generate's pre-check cannot read, an m given to the wrong set,
    an answer the constants do not hold, and a draft's leftovers all fail before the commit, not in the job."""
    text = DECISIONS.read_text(encoding="utf-8") + edit(standin_d023())
    with pytest.raises(AssertionError):
        assert_d023_states_the_registered_values(text)


def test_d023_states_the_registered_values():
    """D-023 holds the mark and the answers line verbatim (each alone on its line), the rule, the counts, m in each
    set's bullet, check's condition counts, the amendment's form, the separation rules, the RM loader's ask, the job,
    its walltime and its cost, and none of the draft's placeholders. It is written with Shresth's answers (spec
    §6.2 step 4): until then this test fails, as the branch expects."""
    assert_d023_states_the_registered_values(DECISIONS.read_text(encoding="utf-8"))
