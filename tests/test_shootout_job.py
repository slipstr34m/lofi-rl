"""The `lofi-shootout` job (docs/SHOOTOUTS.md section 7.7): jobs/shootout.sbatch and the stage runner it calls.

The script follows tests/test_job_scripts.py (name, scratch log, job_env, cost header) and section 7.7's numbers
(partition main, 8 CPU, 16 GB, 20 min: 12 SU/min, at most 240 SU). The runner, `python -m lofi_rl.shootout.job`,
takes the job's ARGS, checks its --leaders as the draw does, writes them all into summary/leaders.json, and runs the
stages in order (the draw, the generator with --levels none, the levels tool, the generator, the clip renders, the
batches); the first stage that fails ends the job before any batch is written. Its registration in
jobs/APPROVED.yaml is left to `make approve`."""
import json
import re
import subprocess
from pathlib import Path

import pytest

from lofi_rl.shootout import job
from lofi_rl.shootout.draw import DrawError, draw, parse_leaders
from lofi_rl.shootout.plan import load_plan

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "jobs" / "shootout.sbatch"
DAY3 = ["--stage", "day3", "--leaders", "keys=fp_fm_ep1/kit=kit_swirly/bass=bass_fp_finger_yr/sax=sax_mtg_tenor",
        "--prior", "13640001+13640002", "--repeats", "K1+R1+S1", "--rerun-s2", "sax_gu_alto", "--cut", "w1-two"]


def _sbatch(name: str) -> str:
    m = re.search(rf"^#SBATCH --{name}=(\S+)$", SCRIPT.read_text(), re.M)
    assert m, name
    return m.group(1)


def test_the_job_script_asks_for_what_section_7_7_prices():
    # Walltime from the Day-1 dry runs (laptop, 2026-10-31: every stage in 87.9-96.7 s over five runs, per-clip
    # faders included, peak memory 3.0 GB): the cluster's slowest render node (x 2.7), 8 CPUs that may be 4 cores'
    # hyperthreads (x 2) and 2 min of start-up come to about 10.7 min, so 20 min is 1.9 times that.
    s = SCRIPT.read_text()
    assert _sbatch("job-name") == "lofi-shootout" and _sbatch("partition") == "main"
    assert _sbatch("cpus-per-task") == "8" and _sbatch("mem") == "16G" and _sbatch("time") == "00:20:00"
    assert _sbatch("output") == "/scratch1/%u/lofi-rl/logs/%x_%j.out"
    cpus, gb, minutes = 8, 16, 20
    rate = cpus + gb / 4                                     # 1 SU per CPU-min, 1 SU per 4 GB-min (CLAUDE.md 4)
    assert rate == 12 and rate * minutes == 240
    assert "Cost: 8 CPU + 4 (16 GB) = 12 SU/min -> at most 240 SU." in s
    assert "summary/stages.json" in s, "the results list names the stage timings"
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
    # from day 2 on the measuring palettes follow the draw: its band, and each matched clip's palette (CD-53)
    assert lines["measure"].endswith(f"lofi_rl.shootout.palettes --plan {job.PLAN} --stage day3 --levels none "
                                     f"--draw {s}/draw.json --out {s}/palettes")
    assert f"--palettes {s}/palettes/measure --stage day3 --draw {s}/draw.json " \
           f"--prior {tmp_path}/runs/shootout-13640001/summary/levels.json" in lines["levels"] \
        and f"--out {s}/levels.json --workers 8" in lines["levels"]
    assert lines["palettes"].endswith(f"lofi_rl.shootout.palettes --plan {job.PLAN} --stage day3 --levels "
                                      f"{s}/levels.json --draw {s}/draw.json --out {s}/palettes")
    assert f"lofi_rl.shootout.render --draw {s}/draw.json --palettes {s}/palettes --out {run}/renders " \
           f"--summary {s} --jobs 8 --workers 1" in lines["render"]
    assert f"batch build --plan {job.PLAN} --stage day3 --renders {run}/renders --out {tmp_path}/inbox " \
           f"--draw {s}/draw.json" in lines["build"]


ARGPARSE = ("unrecognized arguments", "the following arguments are required", "invalid choice", "expected one argument",
            "not allowed with")


def test_every_stage_command_is_one_its_tool_takes(tmp_path, capsys, monkeypatch):
    # Each stage runs a tool's own CLI in a fresh process, so a flag the tool does not take, or a folder one stage
    # writes and the next does not read, would only show on the cluster. 2026-10-31 dry run: the generator was
    # called without --stage, with --draw on the measuring run and --out one level too deep. Here each command
    # line goes through its tool's main() (inputs missing, so each stops at its own checks, never at argparse), and
    # the folders chain: measure -> levels -> palettes -> render -> build.
    from lofi_rl.shootout import batch, levels, palettes, render
    monkeypatch.setenv("LOFI_ASSETS", str(tmp_path / "no-assets"))     # the generator stops at its source check
    a = job._args(["--run", str(tmp_path / "runs" / "shootout-1"), "--inbox", str(tmp_path / "inbox"), "--stage",
                   "day1", "--seed", "0000000000000001"])
    steps = dict(job.commands(a, job.check(a)))
    tools = {"draw": batch, "measure": palettes, "levels": levels, "palettes": palettes, "render": render,
             "build": batch}
    for name, argv in steps.items():
        assert argv[1:3] == ["-m", tools[name].__name__], name
        try:
            tools[name].main(argv[3:])
        except SystemExit:
            pass
        err = capsys.readouterr().err
        assert not any(x in err for x in ARGPARSE), (name, err)
    arg = {name: dict(zip(argv[3:], argv[4:])) for name, argv in steps.items()}
    assert arg["levels"]["--palettes"] == arg["measure"]["--out"] + "/measure", "the measuring palettes' folder"
    assert arg["palettes"]["--levels"] == arg["levels"]["--out"] and arg["render"]["--palettes"] == arg["palettes"]["--out"]
    assert arg["render"]["--draw"] == arg["palettes"]["--draw"] == arg["build"]["--draw"] == arg["levels"]["--draw"]
    assert arg["levels"]["--draw"] == arg["draw"]["--summary"] + "/draw.json", "the levels see every clip drawn"
    assert arg["build"]["--renders"] == arg["render"]["--out"] and "--draw" not in arg["measure"]
    assert {arg[n]["--stage"] for n in ("draw", "measure", "levels", "palettes", "build")} == {"day1"}


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


# --leaders, as each day's draw takes it (SHOOTOUTS.md section 7.7): day 2's K2 plays the K1 top 3 (4 on a tie at
# third place) and its S1 final the S1 top 2 (3 on a tie), so keys takes 3-4 ids and sax 2-3, or 1 when the `final`
# cut drops the final; day 3 plays one instrument per slot.
KEYS3 = "gu_tine_ep+fp_fm_ep1+gu_organ"
KIT_BASS = "kit=kit_swirly/bass=bass_fp_finger_yr"


@pytest.mark.parametrize("stage,leaders,cut,said", [
    # SHOOTOUTS.md 7.7's day-2 form until 2026-10-31: one sax id
    ("day2", f"keys={KEYS3}/{KIT_BASS}/sax=sax_mtg_tenor", "", "--leaders sax: 1 ids given, day2 takes [2, 3]"),
    # job.py's docstring until 2026-10-31: "sax=<id>+<id> from day 2 on" (and three keys)
    ("day3", f"keys=fp_fm_ep1/{KIT_BASS}/sax=sax_mtg_tenor+sax_gu_alto", "",
     "--leaders sax: 2 ids given, day3 takes [1]"),
    ("day3", f"keys={KEYS3}/{KIT_BASS}/sax=sax_mtg_tenor", "", "--leaders keys: 3 ids given, day3 takes [1]"),
    # with the `final` cut the S1 leader is day 2's only sax
    ("day2", f"keys={KEYS3}/{KIT_BASS}/sax=sax_mtg_tenor+sax_gu_alto", "final",
     "--leaders sax: 2 ids given, day2 takes [1]"),
])
def test_a_leader_count_the_draw_refuses_stops_the_job_before_any_stage(tmp_path, capsys, monkeypatch, stage,
                                                                        leaders, cut, said):
    # The draw refuses these at its stage (exit 2), after the job has waited in the queue and started. The runner's
    # own check runs the draw's leader check, after the cuts, so --dry-run and a real run both exit 2 first.
    cuts = [cut] if cut else []
    with pytest.raises(DrawError, match=re.escape(said)):
        draw(load_plan(), stage, seed=1, leaders=parse_leaders(leaders), cuts=cuts)
    args = ["--run", str(tmp_path / "run"), "--inbox", str(tmp_path / "inbox"), "--stage", stage, "--leaders", leaders,
            *(["--cut", cut] if cut else [])]
    assert job.main([*args, "--dry-run"]) == 2 and said in capsys.readouterr().err
    monkeypatch.setattr(job, "find_spec", lambda name: object())
    ran = []
    monkeypatch.setattr(job, "run_step", lambda name, argv: ran.append(name) or 0)
    assert job.main(args) == 2 and ran == [] and said in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("stage,cut,takes", [
    ("day2", "", {(3, 2), (3, 3), (4, 2), (4, 3)}),
    ("day2", "final", {(3, 1), (4, 1)}),
    ("day3", "", {(1, 1)}),
])
def test_the_job_takes_exactly_the_leader_counts_its_draw_takes(tmp_path, stage, cut, takes):
    # (keys ids, sax ids) from 1-4 keys and 1-3 saxes, one kit and one bass: the runner's check and the draw agree.
    plan = load_plan()
    o = plan.band_options
    cuts = [cut] if cut else []
    drawn, checked = set(), set()
    for k in range(1, 5):
        for s in range(1, 4):
            text = f"keys={'+'.join(o['keys'][:k])}/kit={o['kit'][0]}/bass={o['bass'][0]}/sax={'+'.join(o['sax'][:s])}"
            try:
                draw(plan, stage, seed=1, leaders=parse_leaders(text), cuts=cuts)
                drawn.add((k, s))
            except DrawError:
                pass
            a = job._args(["--run", str(tmp_path / "run"), "--inbox", str(tmp_path / "inbox"), "--stage", stage,
                           "--leaders", text, *(["--cut", cut] if cut else [])])
            try:
                job.check(a)
                checked.add((k, s))
            except job.JobError:
                pass
    assert drawn == takes
    assert checked == takes


DOCUMENTED = (ROOT / "docs" / "SHOOTOUTS.md", SCRIPT, ROOT / "src" / "lofi_rl" / "shootout" / "job.py",
              ROOT / "src" / "lofi_rl" / "shootout" / "batch.py", ROOT / "src" / "lofi_rl" / "shootout" / "draw.py")
FORM = re.compile(r"--leaders ([a-z_]+=[^\s'`\]]+)|\b(keys=<id>[^\s'`\]]*)")


def _filled(form: str, plan) -> str:
    """A documented --leaders form with each <id> replaced by the next candidate of its slot."""
    out = []
    for part in form.split("/"):
        slot, _, ids = part.partition("=")
        pool = iter(plan.band_options[slot])
        out.append(slot + "=" + "+".join(next(pool) if i == "<id>" else i for i in ids.split("+")))
    return "/".join(out)


def test_every_documented_leaders_form_is_one_the_draw_and_the_job_take(tmp_path):
    # SHOOTOUTS.md 7.7 gave day 2 one sax, and job.py said "sax=<id>+<id> from day 2 on": a job submitted from either
    # text would wait in the queue and fail at its draw. Every form these files give names its --stage on its line,
    # the draw takes it, and the docs and the job script each give day 2's and day 3's.
    plan = load_plan()
    days: dict[str, set[str]] = {}
    wrong = []
    for path in DOCUMENTED:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in FORM.finditer(line):
                form = m.group(1) or m.group(2)
                stage = re.findall(r"--stage (day[123])\b", line[:m.start()])
                if not stage:
                    wrong.append(f"{path.name}:{n}: {form} names no --stage")
                    continue
                text = _filled(form, plan)
                try:
                    draw(plan, stage[-1], seed=1, leaders=parse_leaders(text))
                    job.check(job._args(["--run", str(tmp_path / "run"), "--inbox", str(tmp_path / "inbox"),
                                         "--stage", stage[-1], "--leaders", text]))
                except (DrawError, job.JobError) as err:
                    wrong.append(f"{path.name}:{n}: {stage[-1]} {form}: {err}")
                    continue
                days.setdefault(path.name, set()).add(stage[-1])
    assert wrong == []
    assert {name: days.get(name) for name in ("SHOOTOUTS.md", "shootout.sbatch", "job.py")} == \
        dict.fromkeys(("SHOOTOUTS.md", "shootout.sbatch", "job.py"), {"day2", "day3"})


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
    timed = json.loads((tmp_path / "run" / "summary" / "stages.json").read_text())
    assert [(s["stage"], s["exit"]) for s in timed["stages"]] == [("draw", 0), ("measure", 0), ("levels", 0),
                                                                 ("palettes", 0), ("render", 3)]
    ran.clear()
    monkeypatch.setattr(job, "run_step", lambda name, argv: ran.append(name) or 0)
    assert job.main(["--run", str(tmp_path / "run2"), "--inbox", str(tmp_path / "inbox"), "--stage", "day1"]) == 0
    assert ran[-1] == "build" and json.loads((tmp_path / "run2" / "summary" / "leaders.json").read_text())["stage"] == "day1"


def test_the_runner_records_each_stages_wall_time(tmp_path, capsys, monkeypatch):
    # The SU estimate of jobs/shootout.sbatch rests on measured stage times (SHOOTOUTS.md §7.7), so every run
    # leaves them in summary/stages.json (pulled by `make pull`) and prints them in the job log.
    monkeypatch.setattr(job, "find_spec", lambda name: object())
    clock = iter(range(0, 100, 5))
    monkeypatch.setattr(job.time, "perf_counter", lambda: float(next(clock)))
    monkeypatch.setattr(job, "run_step", lambda name, argv: 0)
    assert job.main(["--run", str(tmp_path / "run"), "--inbox", str(tmp_path / "inbox"), "--stage", "day1"]) == 0
    doc = json.loads((tmp_path / "run" / "summary" / "stages.json").read_text())
    assert doc["schema"] == "lofi-rl/shootout-stages/1" and doc["jobs"] == 8
    assert [s["stage"] for s in doc["stages"]] == list(job.STEPS)
    assert all(s["seconds"] == 5.0 and s["exit"] == 0 for s in doc["stages"]) and doc["seconds"] == 30.0
    out = capsys.readouterr().out
    assert "shootout: render: done in 5.0 s" in out and "shootout: every stage done in 30.0 s" in out
