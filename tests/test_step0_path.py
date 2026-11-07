"""The path file (lofi_rl.step0.path; docs/design/step0_samples.md §4.1, §6.1) and the step-0 command line
(lofi_rl.step0.cli).

What these tests pin, on temporary trees (the repo's own docs/prereg/D-023_step0_path.txt is written at the code
commit, last, with Shresth's answers, so no test writes it):
- the list: the import closure of lofi_rl.step0 (every module of the package, imports inside functions included, and
  each package's __init__.py), jobs/step0.sbatch, scripts/job_env.sh, scripts/_kit_env.sh, kit.env and the committed
  data files (the palette and its mode-commit source, prompt v2's 9 example files, the streams file); never
  docs/DECISIONS.md or the path file itself;
- the form: D-020_junk_path.txt's "<git blob sha>  <path>" lines, sorted, under # comments; written twice it is
  byte-identical, and two copies of one tree give one text;
- a changed file is caught by blob in a tree (no git) and by git at a commit, a redo's excepted files aside;
- a missing listed file (jobs/step0.sbatch before it exists) is refused, nothing written;
- the command line: every argument the spec gives each subcommand, and a module not built yet is a clear error.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess

import pytest

import lofi_rl.step0 as S
from lofi_rl.step0 import Step0Error, cli
from lofi_rl.step0 import path as P

pytestmark = pytest.mark.laptop   # the git half needs git: never on a compute node (CLAUDE.md §1)

ROOT = P.ROOT
SF2 = "assets/soundfonts/generaluser-gs/GeneralUser-GS.sf2"
EXAMPLES = ["tests/fixtures/loops/example.ls"] + [f"shootouts/loops/{x}.ls" for x in (
    "l1_melancholy_low", "l2_melancholy_mid", "l3_melancholy_high", "l4_wistful_low", "l6_wistful_high",
    "l7_hopeful_low", "l8_hopeful_mid", "l9_hopeful_high")]
JOB_TEXT = "#!/bin/bash\n#SBATCH --job-name=lofi-step0\n# a stand-in for the job script, for these tests\n"
GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="T",
               GIT_COMMITTER_EMAIL="t@e")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git (the laptop)")


def git(repo, *args) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=GIT_ENV, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout


def tree(tmp_path, name: str = "tree", job: bool = True):
    """A copy of what the list reads: src/, the data files (the SoundFont as a small stand-in: the palette is loaded
    without its sources), the env scripts and kit.env, and a stand-in jobs/step0.sbatch."""
    root = tmp_path / name
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel in [*P.data_files(ROOT), "scripts/job_env.sh", "scripts/_kit_env.sh", "kit.env"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel == SF2:
            (root / rel).write_bytes(b"a stand-in for the SoundFont\n")
        else:
            shutil.copy(ROOT / rel, root / rel)
    if job:
        (root / "jobs").mkdir()
        (root / S.JOB).write_text(JOB_TEXT)
    return root


# --------------------------------------------------------------------------------------------- the list
def test_the_list_is_the_closure_the_job_the_env_scripts_and_the_data_files(tmp_path):
    root = tree(tmp_path)
    entries, malformed = P.parse(P.text(root))
    paths = [p for _, p in entries]
    assert malformed == [] and paths == sorted(paths) and len(paths) == len(set(paths))
    step0 = {f"src/lofi_rl/step0/{f.name}" for f in (root / "src/lofi_rl/step0").glob("*.py")}
    assert {"src/lofi_rl/step0/__init__.py", "src/lofi_rl/step0/__main__.py", "src/lofi_rl/step0/cli.py",
            "src/lofi_rl/step0/select.py", "src/lofi_rl/step0/requests.py", "src/lofi_rl/step0/path.py"} <= step0
    closure = set(P.closure(root))
    assert step0 <= closure, "every module of the package"
    assert {"src/lofi_rl/__init__.py", "src/lofi_rl/brief_streams.py", "src/lofi_rl/brief.py",
            "src/lofi_rl/prng.py", "src/lofi_rl/palette.py", "src/lofi_rl/prompt_v2.py",
            "src/lofi_rl/dsl/__init__.py"} <= closure
    assert all(p.startswith(("src/lofi_rl/", "src/kit/")) and p.endswith(".py") for p in closure)
    data = [S.PALETTE, SF2, *EXAMPLES, S.STREAMS_FILE]
    assert sorted(P.data_files(root)) == sorted(data) and len(EXAMPLES) == 9
    assert set(paths) == closure | {S.JOB, "scripts/job_env.sh", "scripts/_kit_env.sh", "kit.env"} | set(data)
    assert S.DECISIONS not in paths and S.PATH_FILE not in paths
    assert all(blob == P.git_blob_sha((root / p).read_bytes()) for blob, p in entries)
    body = P.text(root)
    assert body.isascii() and body.startswith("# D-023's step-0 path") and body.endswith(f"  {paths[-1]}\n")
    assert all(line.startswith("#") or (len(line.split("  ")[0]) == 40 and "  " in line)
               for line in body.splitlines())


def test_the_list_of_this_checkout_holds_the_same_files():
    closure = P.closure(ROOT)
    assert {f"src/lofi_rl/step0/{f.name}" for f in (ROOT / "src/lofi_rl/step0").glob("*.py")} <= set(closure)
    assert all((ROOT / p).is_file() for p in [*closure, *P.data_files(ROOT), *P.RUN_FILES[1:]])
    assert P.data_files(ROOT) == [S.PALETTE, SF2, "shootouts/loops/l1_melancholy_low.ls",
                                  "shootouts/loops/l4_wistful_low.ls", "shootouts/loops/l7_hopeful_low.ls",
                                  "shootouts/loops/l2_melancholy_mid.ls", "tests/fixtures/loops/example.ls",
                                  "shootouts/loops/l8_hopeful_mid.ls", "shootouts/loops/l3_melancholy_high.ls",
                                  "shootouts/loops/l6_wistful_high.ls", "shootouts/loops/l9_hopeful_high.ls",
                                  S.STREAMS_FILE], "prompt_v2.EXAMPLES' order"


def test_imports_inside_functions_and_by_literal_name_are_in_the_closure(tmp_path):
    root = tree(tmp_path)
    before = set(P.closure(root))
    (root / "src/lofi_rl/step0/extra.py").write_text(
        "import importlib\n\n\n"
        "def f():\n    from lofi_rl import grammar_mask\n    return grammar_mask\n\n\n"
        "def g():\n    from ..rl import curriculum\n"
        "    return importlib.import_module('lofi_rl.ui.manifest'), curriculum\n")
    after = set(P.closure(root))
    assert after - before >= {"src/lofi_rl/step0/extra.py", "src/lofi_rl/grammar_mask.py",
                              "src/lofi_rl/rl/curriculum.py", "src/lofi_rl/rl/__init__.py",
                              "src/lofi_rl/ui/manifest.py", "src/lofi_rl/ui/__init__.py"}
    assert "src/lofi_rl/ui/serve.py" not in after, "only what is reached"
    (root / "src/lofi_rl/step0/extra.py").write_text("def f():\n    from lofi_rl import no_such_module\n")
    assert "src/lofi_rl/step0/extra.py" in P.closure(root), "a name that is no module is an attribute"
    (root / "src/lofi_rl/step0/extra.py").write_text("def f():\n    import lofi_rl.no_such_module\n")
    with pytest.raises(Step0Error, match="lofi_rl.no_such_module is imported but has no file"):
        P.closure(root)


@needs_git
def test_the_closure_of_junk_at_d020s_commit_is_d020s_list_and_the_imports_inside_its_functions(tmp_path):
    # D-020's frozen path lists junk's closure as it ran: 51 modules, from d497684. The static closure of junk in that
    # commit's tree is those 51 and what smoke.py imports only inside its functions (the stub, which junk never calls)
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", "d497684^{commit}"], capture_output=True).returncode:
        pytest.skip("commit d497684 is not in this clone")
    old = tmp_path / "d497684"
    old.mkdir()
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "d497684", "src"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(old)], input=archive.stdout, check=True)
    listed = [line.split("  ")[1] for line in (ROOT / "docs/prereg/D-020_junk_path.txt").read_text().splitlines()
              if not line.startswith("#") and line.split("  ")[1].startswith("src/")]
    mine = set(P.closure(old, ["lofi_rl.rl.junk"]))
    assert len(listed) == 51
    assert mine == set(listed) | {"src/lofi_rl/compile/stub.py", "src/lofi_rl/compile/stub_from_loop.py"}


# --------------------------------------------------------------------------------------------- writing it
def test_written_twice_it_is_byte_identical_and_one_tree_gives_one_text(tmp_path):
    root = tree(tmp_path)
    out, changed = P.write(root)
    assert changed and out == root / S.PATH_FILE
    first = out.read_bytes()
    assert P.write(root) == (out, False) and out.read_bytes() == first
    other = tree(tmp_path, "copy")
    assert P.text(other).encode("ascii") == first, "no date, commit or absolute path in it"
    entries, _ = P.parse(first.decode("ascii"))
    assert len(entries) == len(P.listed(root))


def test_a_missing_listed_file_is_refused_and_nothing_written(tmp_path):
    root = tree(tmp_path, job=False)
    with pytest.raises(Step0Error, match="jobs/step0.sbatch is listed but cannot be read"):
        P.write(root)
    assert not (root / S.PATH_FILE).exists()
    assert cli.main(["path", "--root", str(root), "--write"]) == 2 and not (root / S.PATH_FILE).exists()


# --------------------------------------------------------------------------------------------- checking it
def test_a_changed_file_is_caught_by_blob_in_a_tree(tmp_path):
    root = tree(tmp_path)
    P.write(root)
    rec = P.check_tree(root)
    assert rec["blobs_ok"] is True and rec["matched"] == rec["files"] == len(P.listed(root)) and "ok" not in rec
    assert rec["sha256"] == hashlib.sha256((root / S.PATH_FILE).read_bytes()).hexdigest()
    sel = root / "src/lofi_rl/step0/select.py"
    sel.write_text(sel.read_text() + "# an edit\n")
    rec = P.check_tree(root)
    assert rec["blobs_ok"] is False and rec["n_changed"] == 1
    assert rec["changed"][0]["path"] == "src/lofi_rl/step0/select.py"
    (root / "kit.env").unlink()
    rec = P.check_tree(root)
    assert rec["n_missing"] == 1 and rec["missing"] == ["kit.env"] and rec["blobs_ok"] is False
    (root / S.PATH_FILE).write_text((root / S.PATH_FILE).read_text() + "not-a-blob  x.py\n")
    assert P.check_tree(root)["n_malformed"] == 1
    assert P.parse("a" * 40 + "  /etc/passwd\n" + "b" * 40 + "  ../x\n" + "c" * 39 + "  y\n")[0] == []


@needs_git
def test_a_changed_file_is_caught_by_git_at_a_commit(tmp_path):
    root = tree(tmp_path)
    git(root, "init", "-q", "-b", "main")
    P.write(root)
    git(root, "add", "-A")
    git(root, "commit", "-qm", "the code commit, with its path file")
    good = git(root, "rev-parse", "HEAD").strip()
    entries, _ = P.parse((root / S.PATH_FILE).read_text())
    tree_blobs = {line.split("\t")[1]: line.split()[2] for line in git(root, "ls-tree", "-r", "HEAD").splitlines()}
    assert all(tree_blobs[p] == blob for blob, p in entries), "the blobs computed without git are git's"
    rec = P.check_commit(root, good)
    assert rec["ok"] is True and rec["matched"] == rec["files"] == len(entries)
    pal = root / S.PALETTE
    pal.write_text(pal.read_text() + "# an edit\n")
    git(root, "commit", "-qam", "a later commit that edits the palette")
    later = git(root, "rev-parse", "HEAD").strip()
    rec = P.check_commit(root, later)
    assert rec["ok"] is False and rec["n_changed"] == 1 and rec["changed"][0]["path"] == S.PALETTE
    assert P.check_commit(root, later, excepted=[S.PALETTE])["ok"] is True, "a redo's amendment names it (§4.2)"
    assert P.check_commit(root, good)["ok"] is True, "the code commit still holds every listed blob"
    git(root, "rm", "-q", "kit.env")
    git(root, "commit", "-qm", "drop kit.env")
    rec = P.check_commit(root, "HEAD")
    assert rec["ok"] is False and rec["missing"] == ["kit.env"]
    assert cli.main(["path", "--root", str(root), "--commit", good]) == 0
    assert cli.main(["path", "--root", str(root), "--commit", "HEAD"]) == 3
    assert cli.main(["path", "--root", str(root), "--commit", "no-such-rev"]) == 2


def test_git_blob_sha_is_git_hash_objects_id():
    # the ids `git hash-object --stdin` prints for these bytes (tests/test_rl_smoke_check.py's vectors)
    assert P.git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert P.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


# --------------------------------------------------------------------------------------------- the command line
def test_path_subcommand_prints_writes_and_checks(tmp_path, capsys):
    root = tree(tmp_path)
    assert cli.main(["path", "--root", str(root)]) == 0
    assert capsys.readouterr().out == P.text(root) and not (root / S.PATH_FILE).exists(), "printing writes nothing"
    assert cli.main(["path", "--root", str(root), "--check"]) == 2, "no path file yet: bad input"
    assert cli.main(["path", "--root", str(root), "--write"]) == 0 and (root / S.PATH_FILE).is_file()
    assert cli.main(["path", "--root", str(root), "--check"]) == 0
    assert "STEP0_PATH ok" in capsys.readouterr().out
    (root / "kit.env").write_text("changed\n")
    assert cli.main(["path", "--root", str(root), "--check"]) == 3
    assert "STEP0_PATH DIFFERS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["path", "--write", "--check"])


def test_every_subcommand_takes_the_arguments_the_spec_gives_it(tmp_path):
    a = cli.parser().parse_args(["generate", "--run", "R"])
    assert (a.run.name, a.deadline_s, a.chunk, a.hang_s) == ("R", 960.0, 128, None)
    a = cli.parser().parse_args(["generate", "--run", "R", "--deadline-s", "500", "--chunk", "64", "--hang-s", "0"])
    assert (a.deadline_s, a.chunk, a.hang_s) == (500.0, 64, 0.0)
    a = cli.parser().parse_args(["score", "--run", "R"])
    assert (a.run.name, a.from_) == ("R", None)
    a = cli.parser().parse_args(["score", "--from", "OUT", "--run", "REDO"])
    assert (a.run.name, a.from_.name) == ("REDO", "OUT")
    a = cli.parser().parse_args(["render", "--run", "R"])
    assert (a.workers, a.deadline_s, a.cache, a.from_) == (4, 900.0, None, None)
    a = cli.parser().parse_args(["render", "--from", "OUT", "--run", "REDO", "--workers", "2", "--cache", "C"])
    assert (a.from_.name, a.run.name, a.workers, a.cache.name) == ("OUT", "REDO", 2, "C")
    a = cli.parser().parse_args(["check", "RUN", "--redo", "REDO", "--json", "--rescore"])
    assert (a.run.name, a.redo.name, a.json, a.rescore) == ("RUN", "REDO", True, True)
    a = cli.parser().parse_args(["prereg", "RUN", "--redo", "REDO", "--write"])
    assert (a.run.name, a.redo.name, a.write) == ("RUN", "REDO", True)
    a = cli.parser().parse_args(["path", "--write"])
    assert (a.write, a.check, a.commit, a.root) == (True, False, None, ROOT)
    for bad in (["generate", "--run", "R", "--chunk", "0"], ["render", "--run", "R", "--workers", "0"],
                ["generate"], ["check"], ["frobnicate"]):
        with pytest.raises(SystemExit) as e:
            cli.main(bad)
        assert e.value.code == 2


def test_a_subcommand_whose_module_is_not_built_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(cli.COMMANDS, "generate", "no_such_step")
    assert cli.main(["generate", "--run", str(tmp_path / "r")]) == 2
    assert "lofi_rl.step0.no_such_step is not built yet" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()
    assert set(cli.COMMANDS) == {"generate", "score", "render", "check", "prereg", "path"}
    assert all(m in ("generate", "score", "render", "check", "prereg", "path") for m in cli.COMMANDS.values()
               if m != "no_such_step")
