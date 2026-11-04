"""Setting `frozen:` in palettes/lofi_rain_v1.yaml after Shresth's APPROVED line (docs/SHOOTOUTS.md §1, "From votes to
the freeze", step 4; docs/DECISIONS.md D-017; the freeze spec §5.1).

    python -m lofi_rl.freeze.seal [--gate gates/GATE_palette.md] [--palette palettes/lofi_rain_v1.yaml] [--runs runs]
        [--inbox inbox] [--levels-day3 runs/shootout-<day-3 job>/summary/levels.json] [--dry-run]

Four steps. A step that finds problems lists every one and refuses (exit 3). Nothing is written unless all four pass:
1. **His line**, read as `freeze check --sealed` reads it (lofi_rl.freeze.gate.approval_date). The gate file has one
   `## Approval` heading. Under it, exactly one line is `APPROVED by Shresth Rana <yyyy-mm-dd>`, with a real date. The
   heading and the line are found by their text, never by a line number. A second such line refuses, and so does a
   line under the heading that starts as an approval in another form. Other lines are ignored, such as a note that
   his line was given in chat and entered verbatim. The date is not before any of his confirmations, the
   `**Shresth's confirmation, <date>:**` line of each day's section (gate.confirmations, which `freeze check --gate`
   compares with lofi_rl.freeze.PICKS). Nor is it after today, Pacific: the calendar the sign-off job dates its batch
   in (`pacific_today`, lofi_rl.drafts.job.today_pacific, whatever the machine's own time zone), so his line, the
   batch and today are all read on one calendar. A line inside a code fence or an HTML comment is never read
   (lofi_rl.freeze.gate, as kit.gates reads approvals), so an example of the form there neither counts nor refuses;
   a comment opened after a `-->` on one line refuses, since where it ends cannot be read.
2. **The palette he heard.**
   - The file's `frozen` is null.
   - **The latest sign-off job** is the highest job id among `<runs>/drafts-<job>/summary/args.json` in sign-off mode
     (`mode: signoff` or `--signoff` among its `argv`; not a dry run), as lofi_rl.freeze.signoff writes them. Its
     `palette_sha256` is this file's sha256. Its batch name is `b<MMDD>-signoff[-<k>]`. Every stage in its
     `stages.json` exited 0, `verify` and `build` among them. Its batch `<inbox>/<date>/<batch>/` has `manifest.json`
     and a hidden manifest of category `signoff`, and every clip there plays `<palette id>@<sha8>`.
   - A sha256 that only an earlier sign-off job recorded refuses: a §4.5 loop has run since.
   - **The listen passed.** That job and its batch are the latest sign-off listen recorded in
     lofi_rl.freeze.SIGNOFFS (his marks of it are recorded before the seal), and that listen tossed no kept loop
     (`tossed` is empty). A listen with a Toss reopens its loops (docs/FREEZE.md §5), and only a later sign-off job
     that passes can be sealed.
   - His date is not before that batch's date (a Pacific date: the job's `--date`, or its Pacific today).
   - `python -m lofi_rl.freeze check --gate` passes on the file.
3. **The rebuild.** `python -m lofi_rl.freeze build --frozen <his date>` writes a candidate next to the file. It must
   differ from the file in exactly one line, `frozen: null` → `frozen: '<his date>'`.
4. **The check.** `freeze check --sealed --verify --gate` passes on the candidate before it replaces the file, so a
   failure leaves the signed-off file as it was. After the atomic replace, `freeze check --sealed --gate` runs on the
   file in place; if it fails, the signed-off bytes are put back.

The seal prints sha256 A (the signed-off file, the palette of the sign-off renders) and B (the frozen file). With
`--dry-run` it runs every step but writes nothing. Exit codes are the freeze CLI's: 0 sealed (or, with --dry-run, it
would seal), 2 bad input (a missing or unreadable file), 3 refused.

The freeze CLI (lofi_rl.freeze.palette) is called as a subprocess through its command line (§3.2 of the spec), so the
seal depends on nothing else of it. Tests pass `cli=` a fake.

Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from lofi_rl import freeze as F
from lofi_rl.drafts import job as J
from lofi_rl.palette import ROOT, PaletteError, parse_palette

from . import FROZEN, JOBS
from .gate import APPROVAL_HEADING, GateError, approval_date, confirmations

GATE_PATH = ROOT / "gates" / "GATE_palette.md"
SIGNOFF_BATCH_RE = re.compile(r"b\d{4}-signoff(?:-\d+)?")
SIGNOFF_CATEGORY = "signoff"
SIGNOFF_FLAG = "--signoff"
REQUIRED_STAGES = ("verify", "build")
RUN_RE = re.compile(r"drafts-(\d+)")
SHA_RE = re.compile(r"[0-9a-f]{64}")
CLI_TIMEOUT_S = 1800


class SealError(ValueError):
    """The seal refused. `problems` lists every problem the failing step found."""

    def __init__(self, problems: Sequence[str]):
        self.problems = [str(p) for p in problems] or ["refused"]
        super().__init__("; ".join(self.problems))


class BadInput(SealError):
    """A file the seal needs is missing or unreadable (exit 2)."""


# --------------------------------------------------------------------------------------------- the gate file
def approval(text: str) -> date:
    """The date on his APPROVED line, read as `freeze check --sealed` reads it (lofi_rl.freeze.gate.approval_date).
    SealError when there is none, or the gate file's `## Approval` section is not one with one such line."""
    try:
        got = approval_date(text)
    except GateError as err:
        raise SealError([str(err)]) from None
    if got is None:
        raise SealError([f"no line `APPROVED by Shresth Rana <yyyy-mm-dd>` under `{APPROVAL_HEADING}`: his line is "
                         f"his to write, or his chat line entered verbatim"])
    return date.fromisoformat(got)


def confirmation_dates(text: str) -> dict[str, date]:
    """His confirmation date for each day: the `**Shresth's confirmation, <date>:**` line under each day's heading
    (lofi_rl.freeze.gate.confirmations; `freeze check --gate` compares them with lofi_rl.freeze.PICKS)."""
    try:
        got = confirmations(text)
    except GateError as err:
        raise SealError([f"his confirmations: {err}"]) from None
    if not got:
        raise SealError(["the gate file holds none of his confirmations"])
    return {day: date.fromisoformat(d) for day, d in got.items()}


def pacific_today() -> date:
    """Today in the sign-off batch's calendar: lofi_rl.drafts.job.today_pacific, the date the lofi-drafts job (and so
    lofi_rl.freeze.signoff) gives a batch when no `--date` is passed. The process's own time zone plays no part, so on
    a machine set to another zone a line and a batch dated his today still seal, and his tomorrow still refuses."""
    return J.today_pacific()


def check_date(when: date, confirmations: Mapping[str, date], today: date) -> list[str]:
    """His date against his confirmations and today (`pacific_today` unless a test passes another)."""
    out = [f"his APPROVED date {when} is before his {day} confirmation of {d}" for day, d in sorted(confirmations.items())
           if when < d]
    if when > today:
        out.append(f"his APPROVED date {when} is after today ({today})")
    return out


# --------------------------------------------------------------------------------------------- the sign-off jobs
@dataclass(frozen=True)
class Signoff:
    job: int
    summary: Path
    args: Mapping
    sha256: str | None                  # the sign-off palette's sha256 as the job recorded it
    batch: str | None
    date: str | None


def recorded_sha256(args: Mapping) -> str | None:
    """The sign-off palette's sha256 as the job's summary/args.json records it: `palette_sha256`, as
    lofi_rl.freeze.signoff writes it (schema lofi-rl/signoff-args/1). None when it is not there."""
    got = args.get("palette_sha256")
    return got if isinstance(got, str) else None


def _is_signoff(args: Mapping) -> bool:
    """A lofi-drafts run in sign-off mode (`mode: signoff`, or --signoff among its arguments), not a dry run."""
    argv = args.get("argv") if isinstance(args.get("argv"), list) else []
    flagged = any(a == SIGNOFF_FLAG or str(a).startswith(SIGNOFF_FLAG + "=") for a in argv)
    return (args.get("mode") == "signoff" or flagged) and not args.get("dry_run")


def _read_json(path: Path, what: str) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise BadInput([f"{what} {path}: {err}"]) from None
    if not isinstance(doc, dict):
        raise BadInput([f"{what} {path}: not a JSON object"])
    return doc


def signoff_runs(runs: Path) -> list[Signoff]:
    """Every sign-off job under `runs`, by job id."""
    out = []
    for d in sorted(Path(runs).glob("drafts-*")):
        m = RUN_RE.fullmatch(d.name)
        args_path = d / "summary" / "args.json"
        if not m or not args_path.is_file():
            continue
        args = _read_json(args_path, "the drafts job's args")
        if not _is_signoff(args):
            continue
        out.append(Signoff(int(m.group(1)), d / "summary", args, recorded_sha256(args), args.get("batch"),
                           args.get("date")))
    return sorted(out, key=lambda s: s.job)


def latest_signoff(runs: Path, inbox: Path, sha256: str, label: str) -> Signoff:
    """The latest sign-off job, which must have rendered the palette whose bytes hash to `sha256` and labelled every
    clip of its batch `label` (`<palette id>@<sha8>`). SealError listing every problem otherwise."""
    jobs = signoff_runs(runs)
    if not jobs:
        raise SealError([f"no sign-off job under {runs} (a drafts-<job>/summary/args.json in sign-off mode, not a dry "
                         f"run): `make pull` after job B"])
    last = jobs[-1]
    where = f"sign-off job {last.job}"
    problems = []
    if last.sha256 is None or not SHA_RE.fullmatch(last.sha256):
        problems.append(f"{where}: its args.json records no sign-off palette sha256 (palette_sha256), got "
                        f"{last.sha256!r}")
    elif last.sha256 != sha256:
        earlier = [s.job for s in jobs[:-1] if s.sha256 == sha256]
        if earlier:
            problems.append(f"the palette's sha256 {sha256[:12]}... is what earlier sign-off job {earlier[-1]} rendered, "
                            f"not the latest, {last.job} ({last.sha256[:12]}...): the seal takes only the latest")
        else:
            problems.append(f"the palette's sha256 {sha256[:12]}... is not what {where} rendered "
                            f"({last.sha256[:12]}...), nor any sign-off job: it was never signed off")
    if not isinstance(last.batch, str) or not SIGNOFF_BATCH_RE.fullmatch(last.batch):
        problems.append(f"{where}: batch {last.batch!r} is not a sign-off batch (b<MMDD>-signoff[-<k>])")
    try:
        date.fromisoformat(str(last.date))
    except ValueError:
        problems.append(f"{where}: its date {last.date!r} is not YYYY-MM-DD")
    stages_path = last.summary / "stages.json"
    if not stages_path.is_file():
        problems.append(f"{where}: no {stages_path}")
    else:
        stages = _read_json(stages_path, "the job's stages").get("stages")
        stages = stages if isinstance(stages, list) else []
        names = [s.get("stage") for s in stages if isinstance(s, Mapping)]
        failed = [s.get("stage") for s in stages if not isinstance(s, Mapping) or s.get("exit") != 0]
        if failed:
            problems.append(f"{where}: stages {failed} did not exit 0")
        missing = [s for s in REQUIRED_STAGES if s not in names]
        if missing:
            problems.append(f"{where}: no {missing} stage in its stages.json {names}")
    if problems:
        raise SealError(problems)
    folder = Path(inbox) / str(last.date) / last.batch
    hidden_path = folder / "manifest.hidden.json"
    if not (folder / "manifest.json").is_file() or not hidden_path.is_file():
        raise SealError([f"{where}: its batch {folder} has no manifest.json and manifest.hidden.json: `make pull`"])
    hidden = _read_json(hidden_path, "the sign-off batch's hidden manifest")
    if hidden.get("batch") != last.batch:
        problems.append(f"{hidden_path}: batch {hidden.get('batch')!r}, not {last.batch!r}")
    if hidden.get("category") != SIGNOFF_CATEGORY:
        problems.append(f"{hidden_path}: category {hidden.get('category')!r}, not {SIGNOFF_CATEGORY!r}")
    rounds = hidden.get("rounds")
    clips = [c for r in rounds.values() if isinstance(r, Mapping) and isinstance(r.get("clips"), Mapping)
             for c in r["clips"].values()] if isinstance(rounds, Mapping) else []
    if not clips:
        problems.append(f"{hidden_path}: no clips")
    wrong = sorted({str(c.get("palette")) if isinstance(c, Mapping) else repr(c) for c in clips
                    if not isinstance(c, Mapping) or c.get("palette") != label})
    if wrong:
        problems.append(f"{hidden_path}: clips play {wrong}, not {label}")
    if problems:
        raise SealError(problems)
    return last


def listen_problems(job: Signoff, signoffs: Sequence[Mapping] | None = None) -> list[str]:
    """The latest sign-off job found under runs (latest_signoff) against the latest sign-off listen recorded
    (lofi_rl.freeze.SIGNOFFS, unless given): the same job and batch, and no kept loop tossed in it."""
    signoffs = F.SIGNOFFS if signoffs is None else signoffs
    if not signoffs:
        return [f"no sign-off listen is recorded in lofi_rl.freeze.SIGNOFFS: his marks of {job.batch} (job {job.job}) "
                f"are recorded before the seal"]
    last, out = signoffs[-1], []
    if (job.job, job.batch) != (last["job"], last["batch"]):
        out.append(f"the latest sign-off job under runs is {job.job} ({job.batch}), but the latest sign-off listen "
                   f"recorded in lofi_rl.freeze.SIGNOFFS is {last['batch']} (job {last['job']}): his marks of "
                   f"{job.batch} are recorded first")
    if last["tossed"]:
        out.append(f"the latest sign-off listen {last['batch']} (job {last['job']}) tossed {len(last['tossed'])} kept "
                   f"loop{'' if len(last['tossed']) == 1 else 's'} ({', '.join(last['tossed'])}): it did not pass, so "
                   f"nothing is sealed until a later sign-off listen passes (docs/FREEZE.md §5)")
    return out


# --------------------------------------------------------------------------------------------- the freeze CLI
@dataclass(frozen=True)
class CLIResult:
    rc: int
    out: str
    err: str


FreezeCLI = Callable[[Sequence[str]], CLIResult]


def freeze_cli(args: Sequence[str]) -> CLIResult:
    """`python -m lofi_rl.freeze <args>` in a fresh interpreter on this code."""
    src = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    try:
        p = subprocess.run([sys.executable, "-m", "lofi_rl.freeze", *args], capture_output=True, text=True, env=env,
                           cwd=src.parent, timeout=CLI_TIMEOUT_S)
    except subprocess.TimeoutExpired as err:
        return CLIResult(124, "", f"timed out after {err.timeout} s")
    return CLIResult(p.returncode, p.stdout, p.stderr)


def _run(cli: FreezeCLI, args: Sequence[str]) -> CLIResult:
    r = cli(list(args))
    if r.rc != 0:
        tail = [x for x in (r.err.strip().splitlines() + r.out.strip().splitlines()) if x][-12:]
        raise SealError([f"`python -m lofi_rl.freeze {' '.join(args)}` exited {r.rc}"] + tail)
    return r


def one_line_diff(old: str, new: str, when: str) -> list[str]:
    """Problems unless `new` is `old` with its one `frozen: null` line written `frozen: '<when>'`."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    want = ("frozen: null\n", f"frozen: '{when}'\n")
    if len(a) != len(b):
        return [f"the rebuild has {len(b)} lines and the signed-off file {len(a)}; they may differ only in the "
                f"`frozen:` line"]
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diff) != 1:
        shown = ", ".join(str(i + 1) for i in diff[:8]) + (" ..." if len(diff) > 8 else "")
        return [f"the rebuild differs from the signed-off file in {len(diff)} lines ({shown or 'none'}); exactly one, "
                f"{want[0].strip()!r} -> {want[1].strip()!r}, may differ"]
    i = diff[0]
    if (a[i], b[i]) != want:
        return [f"line {i + 1} changes {a[i].rstrip(chr(10))!r} -> {b[i].rstrip(chr(10))!r}; the one change allowed "
                f"is {want[0].strip()!r} -> {want[1].strip()!r}"]
    return []


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically (a temporary file in the same folder, then os.replace)."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------------------------- the seal
@dataclass(frozen=True)
class Sealed:
    date: str            # his date, now the palette's `frozen:`
    a: str               # sha256 of the signed-off file (the sign-off renders' palette)
    b: str               # sha256 of the frozen file
    job: int             # the latest sign-off job
    batch: str           # its batch
    written: bool        # False with dry_run


def seal(*, gate: Path, palette: Path, runs: Path, inbox: Path, levels_day3: Path, cli: FreezeCLI = freeze_cli,
         today: date | None = None, dry_run: bool = False, verify: bool = True,
         signoffs: Sequence[Mapping] | None = None, log: Callable[[str], None] = print) -> Sealed:
    """The four steps of the module doc. SealError (BadInput for a missing file) at the first step that fails, before
    anything is written. `today` defaults to `pacific_today()`, the calendar of the sign-off batch's date; `signoffs`
    to lofi_rl.freeze.SIGNOFFS. `verify=False` leaves `--verify` out of step 4's check; only tests pass it."""
    gate, palette, runs, inbox, levels_day3 = (Path(x) for x in (gate, palette, runs, inbox, levels_day3))
    today = pacific_today() if today is None else today

    # 1. his line
    try:
        text = gate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise BadInput([f"the gate file {gate}: {err}"]) from None
    when = approval(text)
    problems = check_date(when, confirmation_dates(text), today)
    if problems:
        raise SealError(problems)
    log(f"seal: his line under {APPROVAL_HEADING} in {gate}: APPROVED by Shresth Rana {when}")

    # 2. the palette he heard
    try:
        old = palette.read_bytes()
    except OSError as err:
        raise BadInput([f"the palette {palette}: {err}"]) from None
    a = _sha256(old)
    try:
        pal = parse_palette(yaml.safe_load(old.decode("utf-8")), str(palette), a)
    except (UnicodeDecodeError, yaml.YAMLError, PaletteError) as err:
        raise BadInput([f"the palette {palette} does not parse: {err}"]) from None
    if pal.frozen is not None:
        raise SealError([f"{palette} is already frozen ({pal.frozen}): after the freeze it never changes (D-017)"])
    job = latest_signoff(runs, inbox, a, f"{pal.id}@{a[:8]}")
    problems = listen_problems(job, signoffs)
    if problems:
        raise SealError(problems)
    if when < date.fromisoformat(job.date):
        raise SealError([f"his APPROVED date {when} is before the latest sign-off batch, {job.batch} of {job.date}"])
    log(f"seal: {palette} (sha256 A {a}) is what sign-off job {job.job} rendered for {job.batch} ({job.date})")
    r = _run(cli, ["check", "--palette", str(palette), "--gate", str(gate)])
    log(f"seal: freeze check --gate: {r.out.strip().splitlines()[-1] if r.out.strip() else 'exit 0'}")

    # 3. the rebuild, and 4. its check, before anything is written
    if not levels_day3.is_file():
        raise BadInput([f"the Day-3 levels file {levels_day3} is missing (`make pull`): the rebuild needs it"])
    with tempfile.TemporaryDirectory(prefix=".seal-", dir=palette.parent) as tmp:
        candidate = Path(tmp) / palette.name
        _run(cli, ["build", "--levels-day3", str(levels_day3), "--frozen", when.isoformat(), "--out", str(candidate)])
        try:
            new = candidate.read_bytes()
        except OSError as err:
            raise SealError([f"the rebuild wrote no {candidate}: {err}"]) from None
        problems = one_line_diff(old.decode("utf-8"), new.decode("utf-8", errors="replace"), when.isoformat())
        if problems:
            raise SealError(problems)
        b = _sha256(new)
        r = _run(cli, ["check", "--palette", str(candidate), "--gate", str(gate), "--sealed"]
                 + (["--verify"] if verify else []))
        log(f"seal: freeze check --sealed{' --verify' if verify else ''} on the rebuild: "
            f"{r.out.strip().splitlines()[-1] if r.out.strip() else 'exit 0'}")
        log(f"seal: A {a}  (the signed-off file, sign-off job {job.job})")
        log(f"seal: B {b}  (the frozen file, frozen: '{when}')")
        if dry_run:
            log(f"seal: dry run: {palette} is unchanged")
            return Sealed(when.isoformat(), a, b, job.job, job.batch, False)
        if palette.read_bytes() != old:
            raise SealError([f"{palette} changed while the seal ran; nothing was written"])
        os.replace(candidate, palette)
    try:
        _run(cli, ["check", "--palette", str(palette), "--gate", str(gate), "--sealed"])
    except SealError as err:
        _replace(palette, old)
        raise SealError(err.problems + [f"{palette} is back to the signed-off file (sha256 A {a})"]) from None
    if _sha256(palette.read_bytes()) != b:
        raise SealError([f"{palette} does not hash to B {b} after the write"])
    log(f"sealed: {palette} frozen {when} (sha256 B {b})")
    return Sealed(when.isoformat(), a, b, job.job, job.batch, True)


def main(argv: list[str] | None = None, *, cli: FreezeCLI = freeze_cli, today: date | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.freeze.seal", description=__doc__.split("\n\n")[0])
    p.add_argument("--gate", type=Path, default=GATE_PATH, help="the gate file holding his APPROVED line")
    p.add_argument("--palette", type=Path, default=FROZEN, help="the signed-off palette, sealed in place")
    p.add_argument("--runs", type=Path, default=ROOT / "runs", help="the pulled run folders (the sign-off jobs)")
    p.add_argument("--inbox", type=Path, default=ROOT / "inbox", help="the pulled batches (the sign-off batch)")
    p.add_argument("--levels-day3", type=Path, default=None,
                   help=f"Day 3's levels.json for the rebuild (default <runs>/shootout-{JOBS['day3']}/summary/...)")
    p.add_argument("--dry-run", action="store_true", help="every step and check, nothing written")
    a = p.parse_args(argv)
    levels = a.levels_day3 or a.runs / f"shootout-{JOBS['day3']}" / "summary" / "levels.json"
    try:
        seal(gate=a.gate, palette=a.palette, runs=a.runs, inbox=a.inbox, levels_day3=levels, cli=cli, today=today,
             dry_run=a.dry_run)
    except BadInput as err:
        for x in err.problems:
            print(f"error: {x}", file=sys.stderr)
        return 2
    except SealError as err:
        for x in err.problems:
            print(f"refused: {x}", file=sys.stderr)
        print("seal: nothing written" if not a.dry_run else "seal: dry run refused", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
