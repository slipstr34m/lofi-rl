"""The path file docs/prereg/D-023_step0_path.txt (docs/design/step0_samples.md §4.1; D-023): every file the step-0 job
runs or reads, as "<git blob sha>  <path>" lines in docs/prereg/D-020_junk_path.txt's form.

    python -m lofi_rl.step0 path [--root DIR]                 print the list DIR's tree gives (nothing written)
    python -m lofi_rl.step0 path --write [--root DIR]         write it to DIR/docs/prereg/D-023_step0_path.txt
    python -m lofi_rl.step0 path --check [--root DIR]         every file that file lists has its blob in DIR (no git)
    python -m lofi_rl.step0 path --commit REV [--root DIR]    ... at commit REV of the git repo DIR (the laptop)

What it lists (spec §4.1):
- the import closure of lofi_rl.step0: every module of the package and every lofi_rl or kit module they reach by
  static analysis of their imports (inside functions too, and `importlib.import_module` of a literal name), with each
  package's __init__.py;
- jobs/step0.sbatch, scripts/job_env.sh, scripts/_kit_env.sh and kit.env;
- the committed data files those modules read: the frozen palette and its mode-commit sources, prompt v2's example
  files (the sources of prompt_v2.EXAMPLES: shootouts/loops/<id>.ls, and tests/fixtures/loops/example.ls for "example")
  and the streams file, as D-020's list held prompt v1's example, its palette and the SoundFont.
Not listed: the path file itself, and docs/DECISIONS.md, whose D-023 mark generate checks on its own.

A blob sha is git's, sha1(b"blob <size>\\0" + bytes) (`git hash-object`), computed without git: generate checks the list
in the job's own snapshot, where there is no git (CLAUDE.md §1). On the laptop `--commit` reads a commit's blobs with
`git ls-tree`. The text is a function of the tree alone (no date, no commit), so written twice it is byte-identical;
a file the list needs that is missing (jobs/step0.sbatch before it is written) is refused, with nothing written.

Exit codes: 0 done (--check, --commit: every listed file has its blob), 3 a listed file differs or is missing, 2 bad
input. Standard library only. Sound claims: none.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from collections import deque
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import lofi_rl.step0 as S
from lofi_rl.palette import ROOT
from lofi_rl.step0 import Step0Error

PACKAGE = "lofi_rl.step0"
TOP = ("lofi_rl", "kit")                # the import closure follows these packages (src/); the rest is the env's
RUN_FILES = (S.JOB, "scripts/job_env.sh", "scripts/_kit_env.sh", "kit.env")
EXAMPLE_DIR = "shootouts/loops"         # prompt v2's example <id> came from <id>.ls here (prompt_v2's docstring)
V1_EXAMPLE = "tests/fixtures/loops/example.ls"      # ... and "example", prompt v1's, from this file
HEADER = (
    "# D-023's step-0 path (docs/design/step0_samples.md section 4.1): the git blob sha and path of every file the job",
    "# runs or reads: the import closure of lofi_rl.step0, jobs/step0.sbatch and its env scripts, and the committed",
    "# data files. Written by python -m lofi_rl.step0 path --write, last; generate checks it in the job's snapshot.",
)
BLOB = 40                               # hex digits of a blob sha
MAX_LISTED = 10                         # problem examples kept per kind


def git_blob_sha(data: bytes) -> str:
    """The blob id `git hash-object` prints for these bytes: sha1 of b"blob <size>\\0" + the bytes. The repo has no
    .gitattributes, so no filter changes the bytes git hashes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# --------------------------------------------------------------------------------------------- the import closure
def module_file(root: Path, name: str) -> Path | None:
    """src/<name as a path>/__init__.py or src/<name as a path>.py under `root`, or None."""
    base = Path(root) / "src" / Path(*name.split("."))
    for p in (base / "__init__.py", base.with_name(base.name + ".py")):
        if p.is_file():
            return p
    return None


def imports(tree: ast.AST, module: str, is_package: bool) -> list[tuple[str, tuple[str, ...]]]:
    """(module, names) of every import anywhere in a module's tree, function bodies included: `import a.b` gives
    ("a.b", ()), `from a import b, c` ("a", ("b", "c")) with relative imports resolved against `module`, and
    `importlib.import_module("a.b")` of a literal name ("a.b", ())."""
    package = module if is_package else module.rpartition(".")[0]
    out: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, ()) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                if node.level > len(parts):
                    raise Step0Error([f"{module}: a relative import above its top package (level {node.level})"])
                base = ".".join(parts[:len(parts) - node.level + 1] + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            out.append((base, tuple(a.name for a in node.names)))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str) and (
                    (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                    or (isinstance(node.func, ast.Name) and node.func.id in ("import_module", "__import__"))):
            out.append((node.args[0].value, ()))
    return out


def package_modules(root: Path, package: str = PACKAGE) -> list[str]:
    """Every module of a package directory (not recursive): its __init__, __main__ and other .py files."""
    d = Path(root) / "src" / Path(*package.split("."))
    if not (d / "__init__.py").is_file():
        raise Step0Error([f"{d} is not a package (no __init__.py)"])
    return sorted(package if p.name == "__init__.py" else f"{package}.{p.stem}" for p in d.glob("*.py"))


def closure(root: Path = ROOT, seeds: Iterable[str] | None = None) -> list[str]:
    """The import closure of the modules `seeds` (default: every module of lofi_rl.step0; module doc), as
    repo-relative paths, sorted. Step0Error names every import of a lofi_rl or kit module with no file under
    root/src."""
    root = Path(root)
    todo = deque(package_modules(root) if seeds is None else seeds)
    seen: set[str] = set()
    files: set[str] = set()
    problems: list[str] = []
    while todo:
        name = todo.popleft()
        if name in seen:
            continue
        seen.add(name)
        path = module_file(root, name)
        if path is None:
            problems.append(f"{name} is imported but has no file under {root / 'src'}")
            continue
        files.add(path.relative_to(root).as_posix())
        parts = name.split(".")
        todo.extend(".".join(parts[:j]) for j in range(1, len(parts)))      # its packages' __init__.py
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as err:
            problems.append(f"{path}: cannot be read as Python ({err})")
            continue
        for base, names in imports(tree, name, path.name == "__init__.py"):
            if base.split(".")[0] not in TOP:
                continue
            todo.append(base)
            todo.extend(f"{base}.{n}" for n in names if n != "*" and module_file(root, f"{base}.{n}") is not None)
    if problems:
        raise Step0Error(sorted(problems))
    return sorted(files)


# --------------------------------------------------------------------------------------------- the list
def data_files(root: Path = ROOT) -> list[str]:
    """The committed data files the closure reads (module doc): the palette and its mode-commit sources, prompt v2's
    example files and the streams file."""
    from lofi_rl import prompt_v2
    from lofi_rl.palette import PaletteError, load_palette
    try:
        pal = load_palette(Path(root) / S.PALETTE, verify=False, root=Path(root))
    except (OSError, PaletteError) as err:
        raise Step0Error([f"{S.PALETTE}: cannot be read as a palette ({err})"]) from err
    sources = sorted(s.path for s in pal.sources.values() if s.mode == "commit")
    examples = [V1_EXAMPLE if ex_id == "example" else f"{EXAMPLE_DIR}/{ex_id}.ls"
                for ex_id, _ in prompt_v2.EXAMPLES.values()]
    return [S.PALETTE, *sources, *examples, S.STREAMS_FILE]


def listed(root: Path = ROOT) -> list[str]:
    """Every path the path file lists, sorted (bytewise, as git orders a tree's paths)."""
    return sorted(set(closure(root)) | set(RUN_FILES) | set(data_files(root)))


def text(root: Path = ROOT) -> str:
    """The path file's exact text for the tree at `root`: the header, then "<blob>  <path>" a line. Step0Error, with
    every one named, when a file it lists is missing or unreadable."""
    root = Path(root)
    lines, problems = list(HEADER), []
    for rel in listed(root):
        try:
            lines.append(f"{git_blob_sha((root / rel).read_bytes())}  {rel}")
        except OSError as err:
            problems.append(f"{rel} is listed but cannot be read in {root} ({err.strerror or err})")
    if problems:
        raise Step0Error(problems)
    return "\n".join(lines) + "\n"


def write(root: Path = ROOT, out: Path | None = None) -> tuple[Path, bool]:
    """Write the path file (default root/docs/prereg/D-023_step0_path.txt); (its path, whether its bytes changed)."""
    root = Path(root)
    body = text(root).encode("ascii")
    out = root / S.PATH_FILE if out is None else Path(out)
    changed = not (out.is_file() and out.read_bytes() == body)
    if changed:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(out)
    return out, changed


def parse(body: str) -> tuple[list[tuple[str, str]], list[tuple[int, str]]]:
    """(blob, path) of each line but blanks and # comments, and the (line number, text) of every malformed one: a
    line must be a 40-hex blob, two spaces and a relative path inside the tree."""
    entries, malformed = [], []
    for n, line in enumerate(body.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        blob, sep, path = line.partition("  ")
        rel = PurePosixPath(path)
        if not (sep and len(blob) == BLOB and all(c in "0123456789abcdef" for c in blob) and path
                and path == path.strip() and not rel.is_absolute() and ".." not in rel.parts):
            malformed.append((n, line[:200]))
        else:
            entries.append((blob, path))
    return entries, malformed


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as err:
        raise Step0Error([f"cannot read the path file {path} ({getattr(err, 'strerror', None) or err})"]) from err


def _record(entries, malformed, now: dict[str, str | None], excepted: Iterable[str] = ()) -> dict:
    """The comparison of the listed blobs with `now` (path -> its blob, None when absent)."""
    excepted = set(excepted)
    changed = [{"path": p, "listed": b, "now": now[p]} for b, p in entries if now[p] is not None and now[p] != b]
    missing = [p for _, p in entries if now[p] is None]
    bad = [c for c in changed if c["path"] not in excepted] + [{"path": p, "listed": None, "now": None}
                                                               for p in missing if p not in excepted]
    return {"files": len(entries), "matched": len(entries) - len(changed) - len(missing),
            "changed": changed[:MAX_LISTED], "n_changed": len(changed), "missing": missing[:MAX_LISTED],
            "n_missing": len(missing), "malformed": [{"line": n, "text": t} for n, t in malformed[:MAX_LISTED]],
            "n_malformed": len(malformed), "excepted": sorted(excepted),
            "ok": bool(entries) and not malformed and not bad}


def check_tree(root: Path = ROOT, path_file: Path | None = None) -> dict:
    """Does every file the path file lists hash to its blob in the tree at `root`? No git (generate's pre-check, in
    the job's snapshot). The record holds the path file's sha256, and `blobs_ok` (generate.json `path_file`)."""
    root = Path(root)
    pf = root / S.PATH_FILE if path_file is None else Path(path_file)
    body = _read(pf)
    entries, malformed = parse(body)
    now: dict[str, str | None] = {}
    for _, p in entries:
        try:
            now[p] = git_blob_sha((root / p).read_bytes())
        except OSError:
            now[p] = None
    rec = _record(entries, malformed, now)
    ok = rec.pop("ok")
    return {"path_file": str(pf), "sha256": hashlib.sha256(body.encode("ascii")).hexdigest(), **rec, "blobs_ok": ok}


def commit_blobs(repo: Path, commit: str, paths: Iterable[str]) -> dict[str, str | None]:
    """path -> its blob at `commit` in the git repo `repo` (`git ls-tree -r`), None for a path the commit lacks.
    Laptop only (CLAUDE.md §1)."""
    paths = list(paths)
    try:
        p = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", commit, "--", *paths],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as err:
        raise Step0Error([f"git ls-tree {commit} failed in {repo} ({err})"]) from err
    if p.returncode != 0:
        raise Step0Error([f"git ls-tree {commit} in {repo}: {p.stderr.decode(errors='replace').strip()[:300]}"])
    got: dict[str, str] = {}
    for item in p.stdout.decode("utf-8", errors="replace").split("\0"):
        head, tab, name = item.partition("\t")
        if tab and len(head.split()) == 3:
            got[name] = head.split()[2]
    return {path: got.get(path) for path in paths}


def check_commit(repo: Path, commit: str, body: str | None = None, *, excepted: Iterable[str] = ()) -> dict:
    """Does every file the path file lists have its listed blob at `commit` (git, on the laptop)? `body` is the path
    file's text (default: the file in `repo`'s tree). The files in `excepted` may differ or be absent: a redo's
    amendment names them (spec §4.2)."""
    repo = Path(repo)
    body = _read(repo / S.PATH_FILE) if body is None else body
    entries, malformed = parse(body)
    rec = _record(entries, malformed, commit_blobs(repo, commit, [p for _, p in entries]), excepted)
    return {"commit": commit, **rec}


# --------------------------------------------------------------------------------------------- the command
def _report(rec: dict, where: str) -> str:
    parts = [f"{rec['matched']} of {rec['files']} listed files have their blob {where}"]
    if rec["n_changed"]:
        first = rec["changed"][0]
        parts.append(f"{rec['n_changed']} differ (first {first['path']}: listed {first['listed'][:12]}, now "
                     f"{(first['now'] or '-')[:12]})")
    if rec["n_missing"]:
        parts.append(f"{rec['n_missing']} missing (first {rec['missing'][0]})")
    if rec["n_malformed"]:
        parts.append(f"{rec['n_malformed']} malformed lines (first line {rec['malformed'][0]['line']})")
    if rec["excepted"]:
        parts.append(f"excepted: {', '.join(rec['excepted'])}")
    return "; ".join(parts)


def main(a: argparse.Namespace) -> int:
    """The `path` subcommand (lofi_rl.step0.cli)."""
    root = Path(a.root)
    try:
        if a.write:
            out, changed = write(root)
            n = len(parse(out.read_text(encoding="ascii"))[0])
            print(f"{'wrote' if changed else 'unchanged:'} {out}: {n} files")
            return 0
        if a.check or a.commit:
            rec = check_commit(root, a.commit) if a.commit else check_tree(root)
            ok = rec["ok"] if a.commit else rec["blobs_ok"]
            print(f"STEP0_PATH {'ok' if ok else 'DIFFERS'}: "
                  + _report(rec, f"at {a.commit}" if a.commit else f"in {root}"))
            return 0 if ok else 3
        sys.stdout.write(text(root))
        return 0
    except Step0Error as err:
        print(f"error: nothing written or decided:\n{err}", file=sys.stderr)
        return 2
