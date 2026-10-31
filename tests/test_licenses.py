"""Assets (CLAUDE.md §11 rule 8, docs/DECISIONS.md D-012). Every committed file under assets/ is listed in
assets/MANIFEST.yaml with a license, and its sha256 matches where one is pinned. Nothing non-commercial is
committed: NC sources must be mode=fetch, and files over 50 MiB are never committed."""
import hashlib
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = yaml.safe_load((ROOT / "assets" / "MANIFEST.yaml").read_text())["assets"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_every_asset_file_is_listed_licensed_and_pinned():
    listed = {a["path"]: a for a in MANIFEST}
    files = [p for p in (ROOT / "assets").rglob("*") if p.is_file() and "big" not in p.relative_to(ROOT / "assets").parts
             and p.name not in ("MANIFEST.yaml", "ATTRIBUTION.md", "UPSTREAM", "README.md")]
    for p in files:
        rel = str(p.relative_to(ROOT))
        assert rel in listed, f"{rel} is not in assets/MANIFEST.yaml"
        a = listed[rel]
        assert a.get("license"), rel
        if a.get("sha256"):
            assert _sha256(p) == a["sha256"], f"{rel} does not match its pinned sha256"


def test_nothing_non_commercial_or_huge_is_committed():
    for a in MANIFEST:
        if re.search(r"\bNC\b|non-?commercial", a["license"], re.I):
            assert a["mode"] == "fetch" and a["path"].startswith("assets/big/"), f"NC asset must be fetch-only: {a['path']}"
        p = ROOT / a["path"]
        if a["mode"] == "commit" and p.exists():
            assert p.stat().st_size <= 50 * 1024 * 1024, f"{a['path']} is over 50 MiB: fetch it instead"
