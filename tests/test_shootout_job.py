"""The `lofi-shootout` job (docs/SHOOTOUTS.md section 7.7): jobs/shootout.sbatch and the stage runner it calls.

The script follows tests/test_job_scripts.py (name, scratch log, job_env, cost header) and section 7.7's numbers
(partition main, 8 CPU, 16 GB, 30 min: 12 SU/min, at most 360 SU). The runner, `python -m lofi_rl.shootout.job`,
takes the job's ARGS, writes them all into summary/leaders.json, and runs the stages in order (the draw, the
generator with --levels none, the levels tool, the generator, the clip renders, the batches); the first stage that
fails ends the job before any batch is written. Its registration in jobs/APPROVED.yaml is left to `make approve`."""
import json
import re
import subprocess
from pathlib import Path

import pytest

from lofi_rl.shootout import job

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "jobs" / "shootout.sbatch"
DAY3 = ["--stage", "day3", "--leaders", "keys=fp_fm_ep1/kit=kit_swirly/bass=bass_fp_finger_yr/sax=sax_mtg_tenor",
        "--prior", "13640001+13640002", "--repeats", "K1+R1+S1", "--rerun-s2", "sax_gu_alto", "--cut", "w1-two"]


def _sbatch(name: str) -> str:
    m = re.search(rf"^#SBATCH --{name}=(\S+)$", SCRIPT.read_text(), re.M)
    assert m, name
    return m.group(1)


def test_the_job_script_asks_for_what_section_7_7_prices():
    s = SCRIPT.read_text()
    assert _sbatch("job-name") == "lofi-shootout" and _sbatch("partition") == "main"
    assert _sbatch("cpus-per-task") == "8" and _sbatch("mem") == "16G" and _sbatch("time") == "00:30:00"
    assert _sbatch("output") == "/scratch1/%u/lofi-rl/logs/%x_%j.out"
    cpus, gb, minutes = 8, 16, 30
    rate = cpus + gb / 4                                     # 1 SU per CPU-min, 1 SU per 4 GB-min (CLAUDE.md 4)
    assert rate == 12 and rate * minutes == 360
    assert "Cost: 8 CPU + 4 (16 GB) = 12 SU/min -> at most 360 SU." in s
    assert "MODE=scorer source scripts/job_env.sh" in s and "make submit JOB=jobs/shootout.sbatch" in s
    code = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))
    assert '-m lofi_rl.shootout.job' in code and '"$@"' in code and 'echo "SHOOTOUT_OK"' in code
    assert "$REMOTE_SCR/runs/shootout-" in code and "$REMOTE_SCR/inbox" in code
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_the_runner_writes_every_argument_into_leaders_json_and_lists_the_stages_in_order(tmp_path, capsys):
    run = tmp_path / "runs" / "shootout-13640003"
    rc = job.main(["--run", str(run), "--runs", str(tmp_path / "runs"), "--inbox", str(tmp_path / "inbox"),
                   *DAY3, "--dry-run"])
    assert rc == 0
    doc = json.loads((run / "summary" / "leaders.json").read_text())
    assert doc["stage"] == "day3" and doc["prior"] == ["13640001", "13640002"]
    assert doc["leaders"] == {"keys": ["fp_fm_ep1"], "kit": ["kit_swirly"], "bass": ["bass_fp_finger_yr"],
                              "sax": ["sax_mtg_tenor"]}
    assert doc["repeats"] == ["K1", "R1", "S1"] and doc["rerun_s2"] == "sax_gu_alto" and doc["cut"] == ["w1-two"]
    assert doc["prior_runs"] == [str(tmp_path / "runs" / f"shootout-{j}") for j in ("13640001", "13640002")]
    out = capsys.readouterr().out
    steps = [l.split(":", 1)[0] for l in out.splitlines() if re.match(r"^(draw|measure|levels|palettes|render|build):", l)]
    assert steps == ["draw", "measure", "levels", "palettes", "render", "build"]
    lines = dict(l.split(": ", 1) for l in out.splitlines() if ": " in l and l.split(":")[0] in steps)
    s = str(run / "summary")
    assert f"batch draw --plan {job.PLAN} --stage day3 --summary {s}" in lines["draw"]
    assert f"--prior {tmp_path}/runs/shootout-13640001/summary/batches.json " \
           f"{tmp_path}/runs/shootout-13640002/summary/batches.json" in lines["draw"]
    assert "--repeats K1+R1+S1" in lines["draw"] and "--rerun-s2 sax_gu_alto" in lines["draw"]
    assert "--cut w1-two" in lines["draw"] and "--seed" not in lines["draw"]
    assert f"lofi_rl.shootout.palettes --plan {job.PLAN} --levels none --draw {s}/draw.json --out {s}/palettes/measure" \
        in lines["measure"]
    assert f"--palettes {s}/palettes/measure --stage day3 --prior {tmp_path}/runs/shootout-13640001/summary/levels.json" \
        in lines["levels"] and f"--out {s}/levels.json --workers 8" in lines["levels"]
    assert f"--levels {s}/levels.json --draw {s}/draw.json --out {s}/palettes" in lines["palettes"]
    assert f"lofi_rl.shootout.render --draw {s}/draw.json --palettes {s}/palettes --out {run}/renders " \
           f"--summary {s} --jobs 8 --workers 1" in lines["render"]
    assert f"batch build --plan {job.PLAN} --stage day3 --renders {run}/renders --out {tmp_path}/inbox " \
           f"--draw {s}/draw.json" in lines["build"]


@pytest.mark.parametrize("args,match", [
    (["--stage", "day4"], "invalid choice"),
    (["--stage", "day2", "--prior", "13640001+abc"], "job ids"),
    (["--stage", "day2", "--leaders", "keys"], "leaders"),
    (["--stage", "day1", "--repeats", "K1"], "repeats"),
    (["--stage", "day1", "--repeats", "K1/R1"], "repeats"),
])
def test_bad_job_arguments_stop_before_anything_runs(tmp_path, capsys, args, match):
    rc = job.main(["--run", str(tmp_path / "run"), "--inbox", str(tmp_path / "inbox"), *args, "--dry-run"])
    assert rc == 2 and match in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_a_missing_tool_stops_the_job_before_any_stage(tmp_path, capsys, monkeypatch):
    real = job.find_spec
    monkeypatch.setattr(job, "find_spec", lambda name: None if name == "lofi_rl.shootout.levels" else real(name))
    ran = []
    monkeypatch.setattr(job, "run_step", lambda name, argv: ran.append(name) or 0)
    rc = job.main(["--run", str(tmp_path / "run"), "--inbox", str(tmp_path / "inbox"), "--stage", "day1"])
    assert rc == 2 and ran == [] and "lofi_rl.shootout.levels" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_a_batch_already_in_the_inbox_stops_the_job_before_it_renders(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(job, "find_spec", lambda name: object())
    ran = []
    monkeypatch.setattr(job, "run_step", lambda name, argv: ran.append(name) or 0)
    (tmp_path / "inbox" / "2026-11-03" / "b1103-3-sax").mkdir(parents=True)
    rc = job.main(["--run", str(tmp_path / "run"), "--inbox", str(tmp_path / "inbox"), "--stage", "day1"])
    assert rc == 2 and ran == [] and "b1103-3-sax" in capsys.readouterr().err
    assert job.main(["--run", str(tmp_path / "run2"), "--inbox", str(tmp_path / "inbox"), "--stage", "day2",
                     "--leaders", "keys=gu_tine_ep+fp_fm_ep1+gu_organ/kit=kit_swirly/bass=bass_fp_finger_yr/"
                                  "sax=sax_mtg_tenor+sax_gu_alto"]) == 0 and ran[-1] == "build"


def test_the_first_failing_stage_ends_the_job(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(job, "find_spec", lambda name: object())
    ran = []
    monkeypatch.setattr(job, "run_step", lambda name, argv: ran.append(name) or (3 if name == "render" else 0))
    rc = job.main(["--run", str(tmp_path / "run"), "--inbox", str(tmp_path / "inbox"), "--stage", "day1"])
    assert rc == 3 and ran == ["draw", "measure", "levels", "palettes", "render"]
    assert "stopped at render (exit 3)" in capsys.readouterr().err
    ran.clear()
    monkeypatch.setattr(job, "run_step", lambda name, argv: ran.append(name) or 0)
    assert job.main(["--run", str(tmp_path / "run2"), "--inbox", str(tmp_path / "inbox"), "--stage", "day1"]) == 0
    assert ran[-1] == "build" and json.loads((tmp_path / "run2" / "summary" / "leaders.json").read_text())["stage"] == "day1"
