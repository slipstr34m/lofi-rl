"""The seal (lofi_rl.freeze.seal; the freeze spec §5.1): `frozen:` takes the date on Shresth's APPROVED line, and only
that line of palettes/lofi_rain_v1.yaml changes.

The world here is synthetic: a gate file with his three confirmations, a palette standing in for the signed-off file
(dev_gu.yaml under the frozen id), sign-off runs and their batches as the lofi-drafts job writes them, and a fake
freeze CLI that records its calls. The seal refuses without his line, with two, with a malformed or too-early one, with
his line only inside an HTML comment or a code fence, on an already-frozen file and on the palette of an earlier
sign-off job. It accepts his line with a note beside it, and checks that exactly one line changes. Its today is the
Pacific date, the calendar the sign-off job dates its batch in, whatever the machine's zone. Then the same runs on
copies of the real runs, inbox and gate in a temporary folder, with the fake CLI and with the real one (skipped where
the files are not on this machine)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from lofi_rl.freeze import FROZEN, JOBS
from lofi_rl.freeze import gate as G
from lofi_rl.freeze import seal as S
from lofi_rl.freeze import tables as T
from lofi_rl.palette import load_palette

ROOT = Path(__file__).resolve().parents[1]
LINE = "APPROVED by Shresth Rana 2026-11-05"
TODAY = date(2026, 11, 30)
CONFS = ("2026-10-31", "2026-11-01", "2026-11-03")
STAGES = ("check", "palettes", "render", "verify", "build")


def gate_text(approval: str, confs=CONFS) -> str:
    return ("# Gate: palette_v1 freeze (Phase 1)\n\n## Shootout results\n\n"
            f"### Day 1: voted {confs[0]} (batches b1103-1-keys; job {JOBS['day1']})\n\n"
            f"**Shresth's confirmation, {confs[0]}:** \"Yes, use these\". This answered the proposal.\n\n"
            f"### Day 2: voted {confs[1]} (batches b1104-1-groove; job {JOBS['day2']})\n\n"
            f"**Shresth's confirmation, {confs[1]}:** \"Yes, use these\".\n\n"
            f"### Day 3: voted {confs[2]} (batches b1106-1-mix; job {JOBS['day3']})\n\n"
            f"**Shresth's confirmation, {confs[2]}:** \"Yes, use these\", with \"24 below (Recommended)\".\n\n"
            "## How to approve\nAdd one line of your own, exactly in this form:\n\n"
            "    APPROVED by Shresth Rana <yyyy-mm-dd>\n\n## Approval\n" + approval)


def stand_in() -> str:
    """The signed-off palette's stand-in: dev_gu.yaml under the frozen palette's id (only the id line differs)."""
    text = (ROOT / "palettes" / "dev_gu.yaml").read_text(encoding="utf-8")
    assert text.count("\nid: dev_gu\n") == 1 and text.count("\nfrozen: null\n") == 1
    return text.replace("\nid: dev_gu\n", "\nid: lofi_rain_v1\n")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class World:
    """The seal's inputs under one root, as the laptop holds them after `make pull`."""

    def __init__(self, root: Path):
        self.root = root
        self.gate = root / "gates" / "GATE_palette.md"
        self.palette = root / "palettes" / "lofi_rain_v1.yaml"
        self.runs, self.inbox = root / "runs", root / "inbox"
        self.levels = self.runs / f"shootout-{JOBS['day3']}" / "summary" / "levels.json"

    @property
    def sha(self) -> str:
        return _sha(self.palette.read_bytes())

    def kwargs(self, **kw) -> dict:
        return {"gate": self.gate, "palette": self.palette, "runs": self.runs, "inbox": self.inbox,
                "levels_day3": self.levels, "today": TODAY, "log": lambda _line: None, **kw}

    def signoff(self, job: int = 13700001, *, sha: str | None = None, batch: str = "b1105-signoff",
                day: str = "2026-11-05", label: str | None = None, category: str = "signoff",
                stages: list | None = None, record: bool = True, hidden_batch: str | None = None,
                dry_run: bool = False, argv: list | None = None, mode: str | None = "signoff",
                folder: bool = True) -> Path:
        """One lofi-drafts --signoff run's summary (args.json as lofi_rl.freeze.signoff writes it, stages.json) and its
        batch's two manifests."""
        sha = self.sha if sha is None else sha
        summary = self.runs / f"drafts-{job}" / "summary"
        summary.mkdir(parents=True, exist_ok=True)
        args = {"schema": "lofi-rl/signoff-args/1",
                "argv": argv if argv is not None else [
                    "--run", f"/scratch1/u/lofi-rl/runs/drafts-{job}", "--inbox", "/scratch1/u/lofi-rl/inbox",
                    "--jobs", "8", "--drafts", "drafts/v1", "--signoff", "palettes/lofi_rain_v1.yaml"],
                "drafts": "drafts/v1", "palette": "palettes/lofi_rain_v1.yaml", "palette_id": "lofi_rain_v1",
                "date": day, "batch": batch, "jobs": 8, "runs_root": "/scratch1/u/lofi-rl/runs", "dry_run": dry_run}
        if mode:
            args["mode"] = mode
        if record:
            args["palette_sha256"] = sha
        (summary / "args.json").write_text(json.dumps(args, indent=1) + "\n", encoding="utf-8")
        stages = [{"stage": s, "exit": 0, "seconds": 1.0} for s in STAGES] if stages is None else stages
        (summary / "stages.json").write_text(json.dumps({"schema": "lofi-rl/draft-stages/1", "stages": stages,
                                                         "seconds": 5.0}) + "\n", encoding="utf-8")
        if folder:
            d = self.inbox / day / batch
            d.mkdir(parents=True, exist_ok=True)
            label = f"lofi_rain_v1@{sha[:8]}" if label is None else label
            rounds = {f"r{i:08x}": {"kind": "none", "of": None, "anchor_clip": None, "brief_id": f"d0{i}",
                                    "category": category,
                                    "clips": {f"k{i:010x}": {"clip": f"c{i:016x}", "palette": label}}}
                      for i in (1, 2)}
            (d / "manifest.json").write_text(json.dumps({"v": 1, "batch": batch, "mode": "triage", "rounds": [
                {"id": r, "bpm": 81, "clips": [{"id": k, "file": f"{k}.flac", "sha256": "0" * 64}
                                               for k in x["clips"]]} for r, x in rounds.items()]}), encoding="utf-8")
            (d / "manifest.hidden.json").write_text(json.dumps({"v": 1, "batch": hidden_batch or batch,
                                                                "category": category, "rounds": rounds}),
                                                    encoding="utf-8")
        return summary


def make_world(root: Path, approval: str = LINE + "\n", *, palette: str | None = None, confs=CONFS,
               signoff: bool = True) -> World:
    w = World(root)
    w.gate.parent.mkdir(parents=True, exist_ok=True)
    w.gate.write_text(gate_text(approval, confs), encoding="utf-8")
    w.palette.parent.mkdir(parents=True, exist_ok=True)
    w.palette.write_text(stand_in() if palette is None else palette, encoding="utf-8")
    w.levels.parent.mkdir(parents=True, exist_ok=True)
    w.levels.write_text("{}\n", encoding="utf-8")
    if signoff:
        w.signoff()
    return w


def frozen_line(text: str, when: str) -> str:
    assert text.count("\nfrozen: null\n") == 1
    return text.replace("\nfrozen: null\n", f"\nfrozen: '{when}'\n")


class FakeCLI:
    """`python -m lofi_rl.freeze` as the seal calls it. `build` writes --out from the signed-off palette with its frozen
    line set to --frozen, or what `transform(text, date)` makes of it; `check` exits 0 unless `fail(args)` is true."""

    def __init__(self, world: World, transform=None, fail=None):
        self.world, self.transform, self.fail, self.calls = world, transform, fail, []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        if args[0] == "build":
            out, when = Path(args[args.index("--out") + 1]), args[args.index("--frozen") + 1]
            text = self.world.palette.read_text(encoding="utf-8")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(self.transform(text, when) if self.transform else frozen_line(text, when), encoding="utf-8")
            return S.CLIResult(0, f"wrote {out} (sha256 x)\n", "")
        if self.fail and self.fail(args):
            return S.CLIResult(3, "", "problem: the check found a problem\n")
        return S.CLIResult(0, f"ok: {args[2]} (lofi_rain_v1, sha256 x): (a) parses\n", "")

    def shape(self) -> list[list[str]]:
        names = {str(self.world.palette): "PALETTE", str(self.world.gate): "GATE", str(self.world.levels): "L3"}

        def lab(x: str) -> str:
            if x in names:
                return names[x]
            return "CANDIDATE" if "/.seal-" in x and x.endswith("/lofi_rain_v1.yaml") else x
        return [[lab(a) for a in c] for c in self.calls]


SEALED_CALLS = [["check", "--palette", "PALETTE", "--gate", "GATE"],
                ["build", "--levels-day3", "L3", "--frozen", "2026-11-05", "--out", "CANDIDATE"],
                ["check", "--palette", "CANDIDATE", "--gate", "GATE", "--sealed", "--verify"],
                ["check", "--palette", "PALETTE", "--gate", "GATE", "--sealed"]]


def refused(w: World, cli: FakeCLI | None = None, **kw) -> list[str]:
    """The seal's problems; the palette must be exactly as it was."""
    before = w.palette.read_bytes()
    with pytest.raises(S.SealError) as err:
        S.seal(**w.kwargs(cli=cli or FakeCLI(w), **kw))
    assert w.palette.read_bytes() == before
    assert not [p for p in w.palette.parent.iterdir() if p.name != w.palette.name]     # no candidate or temp file left
    return err.value.problems


# --------------------------------------------------------------------------------------------- the seal
def test_the_seal_sets_frozen_to_his_date_and_changes_one_line(tmp_path):
    w = make_world(tmp_path)
    old = w.palette.read_text()
    a = w.sha
    cli, lines = FakeCLI(w), []
    got = S.seal(**w.kwargs(cli=cli, log=lines.append))
    new = w.palette.read_text()
    assert new == frozen_line(old, "2026-11-05")
    assert [i for i, (x, y) in enumerate(zip(old.splitlines(), new.splitlines())) if x != y] == [old.splitlines().index(
        "frozen: null")] and len(old.splitlines()) == len(new.splitlines())
    assert got == S.Sealed("2026-11-05", a, _sha(new.encode()), 13700001, "b1105-signoff", True)
    assert load_palette(w.palette, verify=False).frozen == "2026-11-05"
    assert cli.shape() == SEALED_CALLS
    assert any(f"A {a}" in x for x in lines) and any(f"B {got.b}" in x for x in lines)
    assert [p.name for p in w.palette.parent.iterdir()] == ["lofi_rain_v1.yaml"]


@pytest.mark.parametrize("approval", [
    "His line was given in chat on 2026-11-05 and entered verbatim by Claude.\n" + LINE + "\n",
    "\n" + LINE + "\n\n(Given in chat, 2026-11-05; entered verbatim.)\n",
    LINE + "   \n",                                                    # trailing spaces aside, as check --sealed reads
])
def test_his_line_with_a_note_beside_it_seals(tmp_path, approval):
    w = make_world(tmp_path, approval)
    assert S.seal(**w.kwargs(cli=FakeCLI(w))).written
    assert load_palette(w.palette, verify=False).frozen == "2026-11-05"


@pytest.mark.parametrize("approval, why", [
    ("", "no line `APPROVED by Shresth Rana <yyyy-mm-dd>`"),
    ("\nThe line will come after the listen.\n", "no line `APPROVED by Shresth Rana <yyyy-mm-dd>`"),
    ("APPROVED by Shresth Rana <yyyy-mm-dd>\n", "is not the form"),
    (f"{LINE}\nAPPROVED by Shresth Rana 2026-11-06\n", "2 APPROVED lines"),
    (f"{LINE}\n{LINE}\n", "2 APPROVED lines"),
    ("approved by shresth rana 2026-11-05\n", "is not the form"),
    ("  APPROVED by Shresth Rana 2026-11-05\n", "is not the form"),
    ("APPROVED by Shresth Rana 2026-11-5\n", "is not the form"),
    ("APPROVED by Shresth Rana 2026-02-30\n", "is not the form"),
    (f"{LINE}\nApproved: yes\n", "is not the form"),
    (f"{LINE}\n\n## Approval\n", "2 '## Approval' headings"),
    (f"<!-- the form he writes, for example:\n{LINE}\n-->\n", "no line `APPROVED by Shresth Rana <yyyy-mm-dd>`"),
    (f"~~~~\n```\n{LINE}\n```\n~~~~\n", "no line `APPROVED by Shresth Rana <yyyy-mm-dd>`"),
    (f"````\n```\n{LINE}\n```\n````\n", "no line `APPROVED by Shresth Rana <yyyy-mm-dd>`"),
])
def test_without_one_exact_line_it_refuses_and_runs_nothing(tmp_path, approval, why):
    w = make_world(tmp_path, approval)
    cli = FakeCLI(w)
    problems = refused(w, cli)
    assert any(why in p for p in problems), problems
    assert cli.calls == []


def test_his_line_outside_the_approval_section_does_not_count(tmp_path):
    w = make_world(tmp_path, "")
    w.gate.write_text(w.gate.read_text().replace("## How to approve\n", f"## How to approve\n{LINE}\n"))
    assert any("no line" in p for p in refused(w))


@pytest.mark.parametrize("line, confs, why", [
    ("APPROVED by Shresth Rana 2026-11-02", CONFS, "before his day3 confirmation of 2026-11-03"),
    ("APPROVED by Shresth Rana 2026-10-30", CONFS, "before his day1 confirmation of 2026-10-31"),
    (LINE, ("2026-10-31", "2026-11-06", "2026-11-03"), "before his day2 confirmation of 2026-11-06"),
    ("APPROVED by Shresth Rana 2026-12-01", CONFS, "after today (2026-11-30)"),
])
def test_a_too_early_or_future_date_refuses(tmp_path, line, confs, why):
    w = make_world(tmp_path, line + "\n", confs=confs, signoff=False)
    w.signoff(day=line[-10:])
    cli = FakeCLI(w)
    assert any(why in p for p in refused(w, cli)), why
    assert cli.calls == []


def test_today_is_the_calendar_the_signoff_batch_is_dated_in(tmp_path, monkeypatch, capsys):
    """With no `today` given, the after-today guard reads lofi_rl.drafts.job.today_pacific, the function the sign-off
    job dates its batch with (lofi_rl.freeze.signoff). On that day's boundary a line and a batch both dated his today
    seal, his tomorrow refuses as after today, and his yesterday refuses as before the batch."""
    from lofi_rl.drafts import job as J
    monkeypatch.setattr(J, "today_pacific", lambda: date(2026, 11, 5))
    assert S.pacific_today() == date(2026, 11, 5)
    w = make_world(tmp_path / "today")                                     # his line and the batch: 2026-11-05
    assert S.seal(**w.kwargs(cli=FakeCLI(w), today=None)).written
    w = make_world(tmp_path / "tomorrow", "APPROVED by Shresth Rana 2026-11-06\n", signoff=False)
    w.signoff(day="2026-11-05")
    assert refused(w, today=None) == ["his APPROVED date 2026-11-06 is after today (2026-11-05)"]
    w = make_world(tmp_path / "yesterday", "APPROVED by Shresth Rana 2026-11-04\n", signoff=False)
    w.signoff(day="2026-11-05")
    assert refused(w, today=None) == ["his APPROVED date 2026-11-04 is before the latest sign-off batch, b1105-signoff "
                                      "of 2026-11-05"]
    w = make_world(tmp_path / "cli")                                       # the command line has no other today
    monkeypatch.setattr(J, "today_pacific", lambda: date(2026, 11, 4))
    assert S.main(_argv(w, "--dry-run"), cli=FakeCLI(w)) == 3
    assert "after today (2026-11-04)" in capsys.readouterr().err


def test_pacific_today_whatever_the_machines_zone():
    """In a process whose own zone puts it on another date than Los Angeles, the seal's today is still the Pacific
    date. At any instant UTC+14 or UTC-12 is on another date than Pacific time, so one of the two runs shows it."""
    zoneinfo = pytest.importorskip("zoneinfo")
    try:
        la = zoneinfo.ZoneInfo("America/Los_Angeles")
    except zoneinfo.ZoneInfoNotFoundError:
        pytest.skip("no time-zone database here (today_pacific falls back to the machine's date)")
    code = "import datetime; from lofi_rl.freeze import seal as S; print(S.pacific_today(), datetime.date.today())"
    elsewhere = []
    for zone in ("LOFI-14", "LOFI+12"):                                    # POSIX: UTC+14 and UTC-12, no database
        before = datetime.now(la).date()
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                           env={**os.environ, "TZ": zone, "PYTHONPATH": str(ROOT / "src")})
        after = datetime.now(la).date()
        assert p.returncode == 0, p.stderr
        seal_today, local_today = (date.fromisoformat(x) for x in p.stdout.split())
        assert seal_today in (before, after), (zone, seal_today, before)
        elsewhere.append(local_today != seal_today)
    assert any(elsewhere)                                                  # the process really was on another date


def test_a_date_before_the_latest_signoff_batch_refuses(tmp_path):
    w = make_world(tmp_path, signoff=False)
    w.signoff(day="2026-11-06", batch="b1106-signoff")
    cli = FakeCLI(w)
    assert refused(w, cli) == ["his APPROVED date 2026-11-05 is before the latest sign-off batch, b1106-signoff of "
                               "2026-11-06"]
    assert cli.calls == []


def test_without_his_confirmations_it_refuses(tmp_path):
    w = make_world(tmp_path)
    w.gate.write_text(w.gate.read_text().replace("**Shresth's confirmation, 2026-11-01:**", "**Confirmed:**"))
    assert any("his confirmations" in p and "Day 2" in p for p in refused(w))


def test_an_already_frozen_file_refuses(tmp_path):
    w = make_world(tmp_path, palette=frozen_line(stand_in(), "2026-11-04"), signoff=False)
    w.signoff()
    cli = FakeCLI(w)
    problems = refused(w, cli)
    assert problems == [f"{w.palette} is already frozen (2026-11-04): after the freeze it never changes (D-017)"]
    assert cli.calls == []


def test_the_palette_of_an_earlier_signoff_job_refuses(tmp_path):
    w = make_world(tmp_path, signoff=False)
    w.signoff(13700001, day="2026-11-04", batch="b1104-signoff")                 # rendered this file
    w.signoff(13700002, sha="e" * 64, day="2026-11-05", batch="b1105-signoff-2")  # a later loop's palette
    cli = FakeCLI(w)
    problems = refused(w, cli)
    assert len(problems) == 1 and "earlier sign-off job 13700001" in problems[0] and "latest, 13700002" in problems[0]
    assert cli.calls == []


def test_the_latest_signoff_job_is_the_highest_job_id(tmp_path):
    w = make_world(tmp_path, signoff=False)
    w.signoff(13700009, day="2026-11-05", batch="b1105-signoff-2")              # rendered this file, last
    w.signoff(13700002, sha="e" * 64, day="2026-11-04", batch="b1104-signoff")
    assert S.seal(**w.kwargs(cli=FakeCLI(w))).job == 13700009


def test_a_palette_no_signoff_job_rendered_refuses(tmp_path):
    w = make_world(tmp_path, signoff=False)
    w.signoff(sha="e" * 64)
    assert any("never signed off" in p for p in refused(w))


@pytest.mark.parametrize("kwargs", [{}, {"argv": ["--drafts", "drafts/v1"], "mode": None}, {"dry_run": True}])
def test_without_a_signoff_job_it_refuses(tmp_path, kwargs):
    w = make_world(tmp_path, signoff=False)
    if kwargs:
        w.signoff(**kwargs)                         # a keep3-style run, or a laptop dry run of the sign-off
    assert any("no sign-off job under" in p for p in refused(w))


@pytest.mark.parametrize("kwargs, why", [
    ({"record": False}, "records no sign-off palette sha256 (palette_sha256)"),
    ({"batch": "b1105-keep3"}, "is not a sign-off batch"),
    ({"day": "2026-11"}, "is not YYYY-MM-DD"),
    ({"stages": [{"stage": s, "exit": 3 if s == "verify" else 0} for s in STAGES]}, "stages ['verify'] did not exit 0"),
    ({"stages": [{"stage": s, "exit": 0} for s in STAGES[:3]]}, "no ['verify', 'build'] stage"),
    ({"folder": False}, "has no manifest.json and manifest.hidden.json"),
    ({"category": "drafts"}, "category 'drafts', not 'signoff'"),
    ({"label": "lofi_rain_v1@deadbeef"}, "clips play ['lofi_rain_v1@deadbeef']"),
    ({"label": "draft_day2_rhodes_jazz@0123abcd"}, "not lofi_rain_v1@"),
    ({"hidden_batch": "b1105-signoff-2"}, "batch 'b1105-signoff-2', not 'b1105-signoff'"),
])
def test_a_signoff_record_that_is_not_his_batch_refuses(tmp_path, kwargs, why):
    w = make_world(tmp_path, signoff=False)
    w.signoff(**kwargs)
    cli = FakeCLI(w)
    assert any(why in p for p in refused(w, cli)), why
    assert cli.calls == []


@pytest.mark.parametrize("kwargs", [{"mode": None}, {"argv": ["--drafts", "drafts/v1", "--signoff=x.yaml"],
                                                       "mode": None}, {"argv": []}])
def test_a_signoff_run_is_known_by_its_mode_or_its_flag(tmp_path, kwargs):
    w = make_world(tmp_path, signoff=False)
    w.signoff(**kwargs)
    assert S.seal(**w.kwargs(cli=FakeCLI(w))).written


def test_the_recorded_sha256_is_args_jsons_palette_sha256():
    assert S.recorded_sha256({"palette_sha256": "a" * 64, "palette": "palettes/lofi_rain_v1.yaml"}) == "a" * 64
    assert S.recorded_sha256({"palette": "palettes/lofi_rain_v1.yaml"}) is None
    assert S.recorded_sha256({"palette_sha256": None}) is None


@pytest.mark.parametrize("name, transform, why", [
    ("another line too", lambda t, d: frozen_line(t, d).replace("\nid: lofi_rain_v1\n", "\nid: lofi_rain_v2\n"),
     "differs from the signed-off file in 2 lines"),
    ("another date", lambda t, d: frozen_line(t, "2026-11-06"), "the one change allowed is 'frozen: null'"),
    ("unquoted", lambda t, d: t.replace("\nfrozen: null\n", f"\nfrozen: {d}\n"), "the one change allowed"),
    ("a line more", lambda t, d: frozen_line(t, d) + "# appended\n", "the rebuild has"),
    ("nothing", lambda t, d: t, "differs from the signed-off file in 0 lines"),
    ("another line instead", lambda t, d: t.replace("\nsr: 44100\n", "\nsr: 44100  # frozen\n"),
     "changes 'sr: 44100' -> 'sr: 44100  # frozen'"),
])
def test_the_rebuild_must_differ_in_exactly_the_frozen_line(tmp_path, name, transform, why):
    w = make_world(tmp_path)
    cli = FakeCLI(w, transform=transform)
    problems = refused(w, cli)
    assert any(why in p for p in problems), (name, problems)
    assert [c[0] for c in cli.calls] == ["check", "build"]               # nothing checked or written after


@pytest.mark.parametrize("fails, calls", [
    (lambda a: "--sealed" not in a, 1),                                  # step 2: the signed-off file's check
    (lambda a: "--verify" in a, 3),                                      # step 4, on the rebuild, before the write
])
def test_a_failing_check_writes_nothing(tmp_path, fails, calls):
    w = make_world(tmp_path)
    cli = FakeCLI(w, fail=fails)
    problems = refused(w, cli)
    assert "problem: the check found a problem" in problems and "exited 3" in problems[0]
    assert len(cli.calls) == calls


def test_a_failing_check_in_place_puts_the_signed_off_file_back(tmp_path):
    w = make_world(tmp_path)
    old = w.palette.read_bytes()
    cli = FakeCLI(w, fail=lambda a: "--sealed" in a and "--verify" not in a)
    with pytest.raises(S.SealError) as err:
        S.seal(**w.kwargs(cli=cli))
    assert w.palette.read_bytes() == old
    assert any("back to the signed-off file" in p for p in err.value.problems)
    assert cli.shape() == SEALED_CALLS


def test_a_dry_run_writes_nothing(tmp_path):
    w = make_world(tmp_path)
    old = w.palette.read_bytes()
    cli = FakeCLI(w)
    got = S.seal(**w.kwargs(cli=cli, dry_run=True))
    assert not got.written and got.a == _sha(old) and got.b == _sha(frozen_line(old.decode(), "2026-11-05").encode())
    assert w.palette.read_bytes() == old and cli.shape() == SEALED_CALLS[:3]


def test_missing_inputs_are_bad_input(tmp_path):
    w = make_world(tmp_path)
    w.levels.unlink()
    with pytest.raises(S.BadInput, match="Day-3 levels file"):
        S.seal(**w.kwargs(cli=FakeCLI(w)))
    w.palette.unlink()
    with pytest.raises(S.BadInput, match="the palette"):
        S.seal(**w.kwargs(cli=FakeCLI(w)))


def _argv(w: World, *extra: str) -> list[str]:
    return ["--gate", str(w.gate), "--palette", str(w.palette), "--runs", str(w.runs), "--inbox", str(w.inbox),
            "--levels-day3", str(w.levels), *extra]


def test_the_cli_exit_codes(tmp_path, capsys):
    w = make_world(tmp_path)
    assert S.main(_argv(w, "--dry-run"), cli=FakeCLI(w), today=TODAY) == 0
    assert "seal: dry run" in capsys.readouterr().out and w.palette.read_text() == stand_in()
    assert S.main(_argv(w), cli=FakeCLI(w), today=TODAY) == 0
    out = capsys.readouterr().out
    assert "sealed:" in out and f"B {w.sha}" in out
    assert S.main(_argv(w), cli=FakeCLI(w), today=TODAY) == 3                # now already frozen
    assert "refused:" in capsys.readouterr().err
    w.levels.unlink()
    w.palette.write_text(stand_in())
    assert S.main(_argv(w), cli=FakeCLI(w), today=TODAY) == 2
    assert "error:" in capsys.readouterr().err


def test_the_cli_defaults_are_the_repo_paths(monkeypatch):
    seen = {}
    monkeypatch.setattr(S, "seal", lambda **kw: seen.update(kw))
    assert S.main([]) == 0
    assert {k: seen[k] for k in ("gate", "palette", "runs", "inbox", "levels_day3", "dry_run")} == {
        "gate": ROOT / "gates" / "GATE_palette.md", "palette": FROZEN, "runs": ROOT / "runs", "inbox": ROOT / "inbox",
        "levels_day3": ROOT / "runs" / f"shootout-{JOBS['day3']}" / "summary" / "levels.json", "dry_run": False}
    assert seen["cli"] is S.freeze_cli


# --------------------------------------------------------------------------------------------- the real files
def real_world(tmp: Path, approval: str | None) -> World:
    """Copies of the real gate file, run summaries and batch manifests under `tmp` (skip where `make pull` has not
    brought them), with `approval` as the gate copy's Approval section body (None: as it is). The palette is a copy
    of the frozen palette with `frozen: null` (the signed-off file), or the stand-in when S2's file is absent."""
    need = ["gates/GATE_palette.md", f"runs/drafts-{JOBS['keep3']}/summary/args.json",
            f"runs/shootout-{JOBS['day3']}/summary/levels.json"]
    for rel in need:
        if not (ROOT / rel).is_file():
            pytest.skip(f"{rel} is not on this machine (make pull brings it to the laptop)")
    w = World(tmp)
    for src in [ROOT / need[0], ROOT / need[2], *sorted((ROOT / "runs").glob("drafts-*/summary/*.json")),
                *sorted((ROOT / "inbox").glob("*/*/manifest*.json"))]:
        dst = tmp / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    if approval is not None:
        text = w.gate.read_text(encoding="utf-8")
        (head, end), lines = T.sections(text, G.APPROVAL_HEADING)[0], text.splitlines(keepends=True)
        w.gate.write_text("".join(lines[:head + 1]) + approval + "".join(lines[end:]), encoding="utf-8")
    w.palette.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN.is_file():
        text = FROZEN.read_text(encoding="utf-8")
        frozen = load_palette(FROZEN, verify=False).frozen
        w.palette.write_text(text if frozen is None else text.replace(f"\nfrozen: '{frozen}'\n", "\nfrozen: null\n"),
                             encoding="utf-8")
    else:
        w.palette.write_text(stand_in(), encoding="utf-8")
    return w


def _real_dates() -> tuple[date, date]:
    """(the day before his latest confirmation, two days after it) in the real gate file."""
    if not (ROOT / "gates" / "GATE_palette.md").is_file():
        pytest.skip("gates/GATE_palette.md is not here")
    last = max(date.fromisoformat(d) for d in G.confirmations((ROOT / "gates" / "GATE_palette.md").read_text()).values())
    return last - timedelta(days=1), last + timedelta(days=2)


def test_on_copies_of_the_real_runs_inbox_and_gate(tmp_path):
    early, ok = _real_dates()
    real = FROZEN.read_bytes() if FROZEN.is_file() else None
    w = real_world(tmp_path / "as_is", None)
    if G.approval_date(w.gate.read_text()) is None:      # before his line: the gate file as it stands refuses
        assert any("no line" in p for p in refused(w))
    w = real_world(tmp_path / "early", f"\nAPPROVED by Shresth Rana {early}\n")
    assert any(f"before his day3 confirmation of {early + timedelta(days=1)}" in p for p in refused(w))
    for name, body in {"comment": f"\n<!-- the form he writes, for example:\nAPPROVED by Shresth Rana {ok}\n-->\n",
                       "fence": f"\n~~~~\n```\nAPPROVED by Shresth Rana {ok}\n```\n~~~~\n"}.items():
        w = real_world(tmp_path / name, body)                 # his line only inside a comment or a fence
        w.signoff(99999999, day=ok.isoformat(), batch=f"b{ok:%m%d}-signoff")
        assert any("no line `APPROVED by Shresth Rana" in p for p in refused(w)), name
    w = real_world(tmp_path / "ok", f"\nHis line, given in chat, entered verbatim:\nAPPROVED by Shresth Rana {ok}\n")
    assert (w.runs / f"drafts-{JOBS['keep3']}" / "summary" / "args.json").is_file()    # keep3's run is no sign-off
    if not S.signoff_runs(w.runs):
        assert any("no sign-off job under" in p for p in refused(w))
    w.signoff(99999999, day=ok.isoformat(), batch=f"b{ok:%m%d}-signoff")
    old = w.palette.read_text()
    cli = FakeCLI(w)
    got = S.seal(**w.kwargs(cli=cli))
    assert got.written and got.job == 99999999 and w.palette.read_text() == frozen_line(old, ok.isoformat())
    assert cli.shape()[1] == ["build", "--levels-day3", "L3", "--frozen", ok.isoformat(), "--out", "CANDIDATE"]
    assert (FROZEN.read_bytes() if FROZEN.is_file() else None) == real      # the real palette is untouched


def test_with_the_freeze_cli_on_copies_of_the_real_runs_inbox_and_gate(tmp_path):
    """The seal end to end with S2's real `python -m lofi_rl.freeze` (check --gate, build --frozen, check --sealed),
    on a copy of the frozen palette with `frozen: null`; `--verify` is left out (it needs the fetched assets)."""
    if not FROZEN.is_file():
        pytest.skip(f"{FROZEN.relative_to(ROOT)} is not built yet (S2)")
    _early, ok = _real_dates()
    w = real_world(tmp_path, f"\nAPPROVED by Shresth Rana {ok}\n")
    first = S.freeze_cli(["check", "--palette", str(w.palette), "--gate", str(w.gate)])
    assert first.rc == 0, first.err
    w.signoff(99999999, day=ok.isoformat(), batch=f"b{ok:%m%d}-signoff")
    old, a, real = w.palette.read_text(), w.sha, FROZEN.read_bytes()
    lines = []
    got = S.seal(**w.kwargs(cli=S.freeze_cli, verify=False, log=lines.append))
    new = w.palette.read_text()
    assert got.written and got.a == a and got.b == _sha(new.encode()) != a
    assert new == frozen_line(old, ok.isoformat())
    assert S.freeze_cli(["check", "--palette", str(w.palette), "--gate", str(w.gate), "--sealed"]).rc == 0
    assert any(x.startswith("seal: freeze check --gate: ok:") for x in lines), lines
    assert FROZEN.read_bytes() == real
