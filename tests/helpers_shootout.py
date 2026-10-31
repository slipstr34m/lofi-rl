"""Synthetic shootout assets for the plan, palette and level tests (docs/SHOOTOUTS.md §7.1, §7.4, §7.8). Not a test
module.

- `tiny_sf2`: a RIFF sfbk file whose only chunk is pdta/phdr. palette.sf2_presets reads it, so a palette naming its
  presets passes load_palette(verify=True); FluidSynth could not play it, and nothing here renders it.
- `kit_archive`: what the §7.4 importer writes: derived/kits/<kit>.tar (44.1 kHz mono float32 one-shots, sorted
  members, mtime 0) and derived/kits/<kit>.yaml, its printed `lanes:` and `choke:` blocks.
- `synthetic_base`: shootouts/palettes/base.yaml with every SoundFont replaced by a tiny_sf2 under `assets`.
- `synthetic_levels`: a levels file shaped like the level tool's, with made-up trims and faders.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import LAYERS

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "shootouts" / "plan.yaml"
BASE_PATH = ROOT / "shootouts" / "palettes" / "base.yaml"
FILE_KITS = ("kit_virtuosity", "kit_swirly")
GU_PRESETS = {(0, 4), (0, 16), (0, 32), (0, 65), (0, 66), (128, 32)}
CHOKE = [["ohat", "hat"], ["ohat", "ohat"]]


def tiny_sf2(path: Path, presets) -> str:
    recs = b"".join(struct.pack("<20sHHHIII", f"p{b}_{p}".encode(), p, b, i, 0, 0, 0)
                    for i, (b, p) in enumerate(sorted(presets)))
    recs += struct.pack("<20sHHHIII", b"EOP", 0, 0, len(presets), 0, 0, 0)
    phdr = b"phdr" + struct.pack("<I", len(recs)) + recs
    body = b"sfbk" + b"LIST" + struct.pack("<I", 4 + len(phdr)) + b"pdta" + phdr
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kit_archive(assets: Path, kit: str, shaker_db: float = 0.0) -> str:
    """One decaying click per layer and round robin (1 or 2 per layer), the shaker lane at `shaker_db` (a borrowed
    lane's offset, §7.4). Returns the tar's sha256."""
    folder = assets / "derived" / "kits"
    folder.mkdir(parents=True, exist_ok=True)
    lanes, members = {}, {}
    for li, lane in enumerate(LANES):
        lanes[lane] = {}
        for yi, layer in enumerate(LAYERS):
            shots = []
            for rr in range(1 + (li + yi) % 2):
                name = f"{lane}/{layer}_{rr + 1}.wav"
                buf = io.BytesIO()
                x = (0.1 * (yi + 1) * (1 + rr) * np.exp(-np.arange(2000) / 300.0)).astype(np.float32)
                sf.write(buf, x, 44100, subtype="FLOAT", format="WAV")
                members[name] = buf.getvalue()
                shots.append({"src": kit, "file": name})
            lanes[lane][layer] = shots
        lanes[lane]["pan"] = 0.0
        lanes[lane]["db"] = shaker_db if lane == "shaker" else 0.0
    tar = folder / f"{kit}.tar"
    with tarfile.open(tar, "w", format=tarfile.USTAR_FORMAT) as t:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(members[name]), 0, 0o644
            t.addfile(info, io.BytesIO(members[name]))
    (folder / f"{kit}.yaml").write_text(yaml.safe_dump({"lanes": lanes, "choke": CHOKE}, sort_keys=False))
    return hashlib.sha256(tar.read_bytes()).hexdigest()


def synthetic_base(tmp: Path, assets: Path) -> Path:
    """The committed base with each SoundFont swapped for a tiny_sf2: fetch sources under `assets`, the GU bank at
    an absolute path (mode commit resolves `root / path`, which an absolute path overrides)."""
    doc = yaml.safe_load(BASE_PATH.read_text())
    for src in doc["sources"].values():
        if src["mode"] == "fetch":
            src["sha256"] = tiny_sf2(assets / src["path"], {(0, 0), (0, 1), (0, 2), (0, 3)})
        else:
            gu = tmp / "gu.sf2"
            src["sha256"], src["path"] = tiny_sf2(gu, GU_PRESETS), str(gu)
    base = tmp / "base.yaml"
    base.write_text(yaml.safe_dump(doc, sort_keys=False))
    return base


def synthetic_levels(index_path: Path, base: Path, out: Path) -> Path:
    """A levels file shaped like the level tool's from a measuring index: GU instruments t = 0, others made up."""
    index = json.loads(index_path.read_text())
    variants = {}
    for i, (vid, v) in enumerate(sorted(index["variants"].items())):
        gu = v["source"] == "gu"
        variants[vid] = {"bus": v["bus"], "source": v["source"], "source_sha256": v["source_sha256"],
                         "kit_map_sha256": v["kit_map_sha256"], "t": 0.0 if gu else round(-7.5 + 0.37 * i, 2),
                         "F": round(1.25 + 0.5 * i, 2)}
    out.write_text(json.dumps({"schema": "lofi-rl/shootout-levels/1", "stage": "day1",
                               "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                               "variants": variants}, indent=1))
    return out


def loops_subset(folder: Path, names: tuple[str, ...], src: Path = ROOT / "shootouts" / "loops") -> Path:
    """A loops folder holding some of the reference loops and their briefs."""
    folder.mkdir(parents=True, exist_ok=True)
    briefs = yaml.safe_load((src / "briefs.yaml").read_text())["loops"]
    (folder / "briefs.yaml").write_text(yaml.safe_dump({"loops": {n: briefs[n] for n in names}}, sort_keys=False))
    for n in names:
        (folder / f"{n}.ls").write_text((src / f"{n}.ls").read_text())
    return folder
