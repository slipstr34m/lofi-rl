"""jobs/APPROVED.yaml (CLAUDE.md §4, §11 rule 6). Every entry points at an existing job whose sha still
matches: a changed job needs re-approval. Entries above self_approve_max_su carry Shresth's name. Claude
cannot self-approve a large job type."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import approve_job  # noqa: E402


def test_registry_entries_are_current_and_attributed():
    entries = approve_job.load()
    limit = approve_job.self_approve_max()
    assert limit == 500
    for path, e in entries.items():
        assert (ROOT / path).exists(), f"{path} is registered but missing"
        assert approve_job.job_sha(ROOT / path) == e["sha"], f"{path} changed since approval: re-approve it"
        if float(e["max_su"]) > limit:
            assert e["by"] == "Shresth", f"{path} max_su={e['max_su']} > {limit} needs Shresth's approval"


def test_sha_ignores_comments_and_blank_lines(tmp_path):
    a, b, c = tmp_path / "a.sbatch", tmp_path / "b.sbatch", tmp_path / "c.sbatch"
    a.write_text("#!/bin/bash\n#SBATCH -t 1\n# one\necho hi\n")
    b.write_text("#!/bin/bash\n#SBATCH -t 1\n\n# two, longer\n   # indented\necho hi\n")
    c.write_text("#!/bin/bash\n#SBATCH -t 2\necho hi\n")
    assert approve_job.job_sha(a) == approve_job.job_sha(b) != approve_job.job_sha(c)


def test_claude_cannot_self_approve_a_large_job_type(capsys):
    rc = approve_job.main(["jobs/tools.sbatch", "--max-su", "501", "--by", "Claude"])
    assert rc == 2 and "needs --by Shresth" in capsys.readouterr().err
