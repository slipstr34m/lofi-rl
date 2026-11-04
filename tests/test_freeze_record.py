"""The freeze tests under a recorded-shaped record (tests/helpers_record.py RECORDED; docs/FREEZE.md §3.6, §5.2).

Recording Shresth's redo picks and his next sign-off listen adds REDOS entries, their jobs in JOBS and SIGNOFFS
entries to lofi_rl.freeze. The freeze tests must stay green through that: a synthetic world pins the record it was
written for, and a test of the real record reads it as it stands. So representative tests of each freeze test module
run here in a pytest process of their own, with RECORDED (two redo rounds, a second listen that passed) put on
lofi_rl.freeze before any test module is imported, JOBS updated in place, as a session that records them leaves the
constants. They must all run and pass. Nothing is rendered for real and nothing is claimed about sound."""
import os
import subprocess
import sys
from pathlib import Path

import helpers_record as R
from lofi_rl import freeze as F
from lofi_rl.freeze import gate as G

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE = [
    "tests/test_freeze_gate.py",
    "tests/test_freeze_tables.py::test_every_round_in_serving_order_with_its_kind_and_tiles",
    "tests/test_freeze_tables.py::test_a_redo_rounds_job_is_published_as_keep3s_is",
    "tests/test_freeze_tables.py::test_it_prints_the_section_after_his_line",
    "tests/test_freeze_tables.py::test_write_puts_the_section_right_after_the_approval_section",
    "tests/test_freeze_signoff.py::test_the_job_runs_every_stage_and_builds_one_triage_batch",
    "tests/test_freeze_signoff.py::test_the_check_and_palettes_record_what_they_found",
    "tests/test_freeze_signoff.py::test_constants_that_disagree_with_the_file_stop_the_job_at_the_check",
    "tests/test_freeze_signoff_redo.py::test_the_worlds_record_keeps_its_rules_and_its_gate_file_confirms_it",
    "tests/test_freeze_checks.py::test_expectations_refuse_what_its_source_did_not_render",
    "tests/test_freeze_checks.py::test_a_tossed_loop_still_kept_refuses_in_expect",
    "tests/test_freeze_checks.py::test_load_expectations_refuses_a_file_that_is_not_one",
    "tests/test_freeze_checks.py::test_expect_writes_schema_2_with_two_sources_and_the_redo_files_own_values",
    "tests/test_freeze_checks.py::test_check_i_names_each_way_the_file_and_the_constants_differ",
    "tests/test_freeze_checks.py::test_the_expect_command_checks_the_record_before_it_writes",
    "tests/test_freeze_seal.py::test_the_seal_sets_frozen_to_his_date_and_changes_one_line",
    "tests/test_freeze_seal.py::test_the_seal_refuses_after_a_signoff_with_tossed_loops",
    "tests/test_freeze_seal.py::test_the_seal_takes_only_the_latest_recorded_listen",
    "tests/test_freeze_seal.py::test_the_latest_recorded_listen_decides_with_the_real_record",
    "tests/test_freeze_seal.py::test_the_seal_reads_lofi_rl_freeze_signoffs",
    "tests/test_freeze_seal.py::test_a_malformed_listen_record_refuses_by_name_and_runs_nothing",
    "tests/test_freeze_seal.py::test_the_cli_exit_codes",
]
CODE = """\
import sys
import helpers_record as R
from lofi_rl import freeze as F
R.put(R.RECORDED)
print(f"record: {len(F.REDOS)} redo rounds ({', '.join(r['key'] for r in F.REDOS)}), {len(F.SIGNOFFS)} sign-off "
      f"listens, JOBS {sorted(F.JOBS)}", flush=True)
import pytest
sys.exit(pytest.main(sys.argv[1:]))
"""


def test_the_recorded_shape_keeps_every_rule_and_replaces_the_four():
    """RECORDED is a record a session could leave: its rules hold, b1103-signoff and the jobs of 2026-11-03 start it,
    and no kept loop of it is tossed (redo1 kept d01, d06 and d10, redo2 d07)."""
    rec = {k.lower(): v for k, v in R.RECORDED.items()}
    assert G.records_problems(**rec) == []
    assert rec["signoffs"][0] == R.LISTEN and {k: rec["jobs"][k] for k in R.TODAY["JOBS"]} == R.TODAY["JOBS"]
    assert F.tossed_kept(redos=rec["redos"], signoffs=rec["signoffs"]) == [] and not rec["signoffs"][-1]["tossed"]
    assert {b: key for b, (_f, key) in F.kept_rounds(redos=rec["redos"]).items() if key != "keep3"} == {
        "d01": "keep3_redo1", "d06": "keep3_redo1", "d07": "keep3_redo2", "d10": "keep3_redo1"}


def test_representative_freeze_tests_pass_under_a_recorded_record(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    p = subprocess.run([sys.executable, "-c", CODE, "-q", "-p", "no:cacheprovider", "--basetemp",
                        str(tmp_path / "inner"), *REPRESENTATIVE], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=900)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-6000:]
    assert out.startswith("record: 2 redo rounds (keep3_redo1, keep3_redo2), 2 sign-off listens, JOBS ['day1', "
                          "'day2', 'day3', 'keep3', 'keep3_redo1', 'keep3_redo2']"), out[:400]
    last = [x for x in p.stdout.splitlines() if x.strip()][-1]
    assert " passed" in last and not any(w in last for w in ("failed", "error", "skipped", "deselected")), last
