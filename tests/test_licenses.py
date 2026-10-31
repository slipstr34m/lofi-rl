"""Assets (CLAUDE.md §11 rule 8, docs/DECISIONS.md D-012). Every committed file under assets/ is listed in
assets/MANIFEST.yaml with a licence, and its sha256 matches where one is pinned. Licences go through the palette's
allow-list, `lofi_rl.palette.licence_problem` (CC0, public domain, permissive, CC BY), so the manifest and the
palettes agree on what the public repo may hold:
- a mode-commit entry needs a licence the allow-list lets the repo commit. The one carve-out is a licence
  document (LICENSE, LICENCE or COPYING) whose licence is "the license text itself", committed in the same
  folder as an asset the allow-list accepts;
- an entry whose licence the allow-list refuses (non-commercial, restricted or unrecognized) is mode fetch and
  lives under assets/big/ (gitignored), so nothing non-commercial is ever committed;
- nothing over 50 MiB is committed."""
import hashlib
import re
from pathlib import Path

import pytest
import yaml

from lofi_rl.palette import licence_problem

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = yaml.safe_load((ROOT / "assets" / "MANIFEST.yaml").read_text())["assets"]
MAX_COMMIT = 50 * 1024 * 1024
UNLISTED = ("MANIFEST.yaml", "ATTRIBUTION.md", "UPSTREAM", "README.md")
LICENCE_TEXT = "the license text itself"
LICENCE_DOC = re.compile(r"(?:LICEN[CS]E|COPYING)(?:[-_][A-Za-z0-9]+)*(?:\.(?:txt|md|rst|html?))?", re.I)
ASSET = "assets/soundfonts/x/bank.sf2"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_problems(entries: list[dict], root: Path = ROOT) -> list[str]:
    """Files under assets/ (outside big/) that are unlisted, unlicensed or changed since their sha256 was pinned."""
    listed = {a["path"]: a for a in entries}
    problems = []
    for p in sorted((root / "assets").rglob("*")):
        if not p.is_file() or "big" in p.relative_to(root / "assets").parts or p.name in UNLISTED:
            continue
        rel = p.relative_to(root).as_posix()
        a = listed.get(rel)
        if a is None:
            problems.append(f"{rel} is not in assets/MANIFEST.yaml")
        elif not a.get("license"):
            problems.append(f"{rel} has no license")
        elif a.get("sha256") and _sha256(p) != a["sha256"]:
            problems.append(f"{rel} does not match its pinned sha256")
    return problems


def refusal(licence: str) -> str | None:
    """Why the allow-list refuses to let the repo commit a file under this licence, or None if it may.

    Only the mode verdict counts: a CC BY licence also needs its attribution, which D-012 puts in the ATTRIBUTION
    file rather than in the manifest."""
    problem = licence_problem(licence, "commit")
    return problem[1] if problem is not None and problem[0] == "mode" else None


def is_licence_text(entry: dict, entries: list[dict]) -> bool:
    """The carve-out: the licence document of an asset the repo commits, kept in the asset's folder (as the
    GeneralUser GS licence asks). Its "licence" is the text itself, which the allow-list cannot name."""
    path = Path(entry["path"])
    if (entry.get("license") or "").strip().lower() != LICENCE_TEXT or not LICENCE_DOC.fullmatch(path.name):
        return False
    return any(other is not entry and other.get("mode") == "commit" and Path(other["path"]).parent == path.parent
               and refusal(other.get("license") or "") is None for other in entries)


def entry_problems(entries: list[dict], root: Path = ROOT) -> list[str]:
    """Manifest entries that break a licence or size rule. A missing licence is refused like an unknown one."""
    problems = []
    for a in entries:
        path, licence, mode = a["path"], a.get("license") or "", a.get("mode")
        if mode not in ("commit", "fetch"):
            problems.append(f"{path}: mode {mode!r} is not commit or fetch")
        why = refusal(licence)
        if why and not (mode == "fetch" and path.startswith("assets/big/")) and not is_licence_text(a, entries):
            problems.append(f"{path}: {why}, under assets/big/")
        p = root / path
        if mode == "commit" and p.exists() and p.stat().st_size > MAX_COMMIT:
            problems.append(f"{path} is over 50 MiB: fetch it instead")
    return problems


def _entry(path: str, licence: str, mode: str = "commit") -> dict:
    return {"path": path, "sha256": None, "license": licence, "mode": mode}


# ------------------------------------------------------------------------------------- the repo's assets
def test_every_asset_file_is_listed_licensed_and_pinned():
    assert file_problems(MANIFEST) == []


def test_every_manifest_entry_follows_the_licence_and_size_rules():
    assert entry_problems(MANIFEST) == []


def test_the_bank_licence_text_is_the_carve_out():
    texts = [a for a in MANIFEST if a["license"] == LICENCE_TEXT]
    assert [a["path"] for a in texts] == ["assets/soundfonts/generaluser-gs/LICENSE.txt"]
    assert entry_problems([a for a in MANIFEST if a not in texts]) == [], "the bank is committable on its own"
    assert entry_problems(texts) != [], "a licence text with no committed asset beside it is no carve-out"


# --------------------------------------------------------------------------------------- each rule bites
@pytest.mark.parametrize("licence", ["All rights reserved", "personal use only", "free to use, not for redistribution",
                                     "CC BY-SA 4.0", "CC BY-ND 4.0", "GPL-3.0-or-later", "unknown", LICENCE_TEXT])
def test_a_committed_entry_needs_a_licence_on_the_allow_list(licence, tmp_path):
    # The manifest used to check only an NC regex, so every one of these could be committed (D-012 allows none).
    assert entry_problems([_entry(ASSET, licence)], tmp_path) == [
        f"{ASSET}: licence {licence!r} is not one D-012 lets the repo commit (CC0, public domain, permissive, CC BY): "
        f"make the source mode fetch, under assets/big/"]
    assert entry_problems([_entry("assets/big/x/bank.sf2", licence, "fetch")], tmp_path) == []
    assert entry_problems([_entry(ASSET, licence, "fetch")], tmp_path) != [], "fetched files live in assets/big/"


@pytest.mark.parametrize("licence", ["CC0 1.0", "public domain", "MIT", "BSD-3-Clause", "Apache-2.0", "CC BY 4.0",
                                     "Creative Commons Attribution 3.0 Unported", MANIFEST[0]["license"]])
def test_a_committed_entry_with_an_allow_listed_licence_passes(licence, tmp_path):
    assert entry_problems([_entry(ASSET, licence)], tmp_path) == []


@pytest.mark.parametrize("licence", ["CC-BY-NC-4.0", "CC BY-NC-SA 3.0", "free for non-commercial use",
                                     "free for non commercial use", "noncommercial"])
def test_non_commercial_entries_are_fetch_only_under_assets_big(licence, tmp_path):
    assert entry_problems([_entry("assets/big/x/kit.sf2", licence, "fetch")], tmp_path) == []
    for path, mode in (("assets/big/x/kit.sf2", "commit"), (ASSET, "fetch"), (ASSET, "commit")):
        assert entry_problems([_entry(path, licence, mode)], tmp_path) == [
            f"{path}: a non-commercial source must be mode fetch (never committed), under assets/big/"], (path, mode)


@pytest.mark.parametrize("name", ["LICENSE.txt", "LICENSE", "LICENCE.md", "COPYING", "license-samples.txt"])
def test_a_licence_text_may_be_committed_beside_its_asset(name, tmp_path):
    asset, text = _entry(ASSET, "CC0 1.0"), _entry(f"assets/soundfonts/x/{name}", LICENCE_TEXT)
    assert entry_problems([asset, text], tmp_path) == []


@pytest.mark.parametrize("entries", [
    [_entry(ASSET, "CC0 1.0"), _entry("assets/soundfonts/x/notes.txt", LICENCE_TEXT)],        # not a licence document
    [_entry(ASSET, "CC0 1.0"), _entry("assets/soundfonts/x/bank2.sf2", LICENCE_TEXT)],
    [_entry(ASSET, "CC0 1.0"), _entry("assets/soundfonts/x/LICENSE.txt.wav", LICENCE_TEXT)],
    [_entry(ASSET, "CC0 1.0"), _entry("assets/soundfonts/y/LICENSE.txt", LICENCE_TEXT)],       # no asset in its folder
    [_entry(ASSET, "CC0 1.0", "fetch"), _entry("assets/soundfonts/x/LICENSE.txt", LICENCE_TEXT)],  # asset not committed
    [_entry(ASSET, "All rights reserved"), _entry("assets/soundfonts/x/LICENSE.txt", LICENCE_TEXT)],  # nor committable
    [_entry("assets/big/x/kit.sf2", "CC-BY-NC-4.0", "fetch"), _entry("assets/big/x/LICENSE", LICENCE_TEXT)],
])
def test_the_licence_text_carve_out_covers_nothing_else(entries, tmp_path):
    problems = entry_problems(entries, tmp_path)
    text = entries[1]["path"]
    assert any(p.startswith(f"{text}: licence 'the license text itself' is not one D-012") for p in problems), problems


@pytest.mark.parametrize("entry, problem", [
    ({"path": ASSET, "sha256": None, "license": "CC0 1.0", "mode": "Commit"},
     f"{ASSET}: mode 'Commit' is not commit or fetch"),
    ({"path": ASSET, "sha256": None, "license": "CC0 1.0"}, f"{ASSET}: mode None is not commit or fetch"),
    ({"path": ASSET, "sha256": None, "mode": "commit"}, f"{ASSET}: licence '' is not one D-012 lets the repo commit"),
])
def test_a_missing_or_misspelled_mode_or_licence_fails_closed(entry, problem, tmp_path):
    # A mode typo would otherwise skip the size rule, which applies to mode commit.
    problems = entry_problems([entry], tmp_path)
    assert len(problems) == 1 and problems[0].startswith(problem), problems


def test_nothing_over_50_mib_is_committed(tmp_path):
    big = tmp_path / ASSET
    big.parent.mkdir(parents=True)
    with big.open("wb") as f:
        f.truncate(MAX_COMMIT + 1)                              # sparse: nothing is written
    assert entry_problems([_entry(ASSET, "CC0 1.0")], tmp_path) == [f"{ASSET} is over 50 MiB: fetch it instead"]
    with big.open("r+b") as f:
        f.truncate(MAX_COMMIT)
    assert entry_problems([_entry(ASSET, "CC0 1.0")], tmp_path) == []


def test_an_unlisted_unlicensed_or_changed_file_is_reported(tmp_path):
    folder = tmp_path / "assets" / "x"
    (tmp_path / "assets" / "big").mkdir(parents=True)
    folder.mkdir()
    for name in ("a.sf2", "b.sf2", "c.sf2", "d.sf2", "README.md"):
        (folder / name).write_bytes(name.encode())
    (tmp_path / "assets" / "big" / "e.sf2").write_bytes(b"fetched, never listed")
    pinned = hashlib.sha256(b"a.sf2").hexdigest()
    entries = [{"path": "assets/x/a.sf2", "sha256": pinned, "license": "CC0 1.0", "mode": "commit"},
               {"path": "assets/x/b.sf2", "sha256": None, "license": "", "mode": "commit"},
               {"path": "assets/x/c.sf2", "sha256": pinned, "license": "CC0 1.0", "mode": "commit"}]
    assert file_problems(entries, tmp_path) == ["assets/x/b.sf2 has no license",
                                                "assets/x/c.sf2 does not match its pinned sha256",
                                                "assets/x/d.sf2 is not in assets/MANIFEST.yaml"]
