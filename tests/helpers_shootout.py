"""Synthetic shootout assets and fake renders for the shootout tests (docs/SHOOTOUTS.md §7.1, §7.4, §7.6, §7.8). Not a
test module.

- `tiny_sf2`: a SoundFont written by the converter's writer (lofi_rl.convert.sf2): one 64-point sample, one
  instrument zone over every key and velocity, and one preset per (bank, prog) asked for. palette.sf2_presets and
  palette.sf2_coverage read it, so a palette naming its presets passes load_palette(verify=True), coverage check
  (SHOOTOUTS.md §7.5) included; nothing here renders it.
- `kit_archive`: what the §7.4 importer writes: derived/kits/<kit>.tar (44.1 kHz mono float32 one-shots, sorted
  members, mtime 0) and derived/kits/<kit>.yaml, its printed `lanes:` and `choke:` blocks, or with `printed=True`
  the whole document `python -m lofi_rl.convert.kit` prints (its report as comments, then `sources`, `choke` and
  `lanes`).
- `synthetic_base`: shootouts/palettes/base.yaml with every SoundFont replaced by a tiny_sf2 under `assets`.
- `synthetic_levels`: a levels file shaped like the level tool's, with made-up trims and faders, and a made-up clip
  fader (CD-52) for every variant on every reference loop and on every loop's sour copy.
- `fake_flac`, `fake_render`, `fake_renders`: what `lofi_rl.shootout.render` writes and the batch builder reads, with
  no audio: `record.json` (the same keys as a real record, `tests/test_shootout_e2e.py` compares them),
  `manifest.json` (n and audio_sha256) and `render.flac`, a FLAC STREAMINFO header with no audio frames. The rating
  page's checks read only that header and the file's sha256, so batches built from them pass
  `python -m lofi_rl.ui.manifest check`. A clip whose draw palette names a `level` (CD-52) gets the level record a
  real render carries (`fake_level`). Never music, never committed.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from lofi_rl.compile.timeline import loop_samples
from lofi_rl.convert.sf2 import Instrument, Preset, Sample, Zone, sf2_bytes
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import LAYERS
from lofi_rl.shootout.render import RECORD_KEYS, RECORD_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "shootouts" / "plan.yaml"
BASE_PATH = ROOT / "shootouts" / "palettes" / "base.yaml"
FILE_KITS = ("kit_virtuosity", "kit_swirly")
GU_PRESETS = {(0, 4), (0, 16), (0, 32), (0, 65), (0, 66), (128, 32)}
CHOKE = [["ohat", "hat"], ["ohat", "ohat"]]


def tiny_sf2(path: Path, presets) -> str:
    """A SoundFont whose presets each play one instrument zone covering keys 0-127 at velocities 0-127."""
    sample = Sample("tick", np.arange(64, dtype=np.int32) * 256, 44100)
    inst = Instrument("all", (Zone(sample="tick"),))
    banks = [Preset(f"p{b}_{p}", b, p, (Zone(instrument="all"),)) for b, p in sorted(presets)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sf2_bytes(name="tiny", presets=banks, instruments=[inst], samples=[sample]))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kit_archive(assets: Path, kit: str, shaker_db: float = 0.0, *, printed: bool = False) -> str:
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
    sha = hashlib.sha256(tar.read_bytes()).hexdigest()
    if not printed:
        (folder / f"{kit}.yaml").write_text(yaml.safe_dump({"lanes": lanes, "choke": CHOKE}, sort_keys=False))
        return sha
    source = {"path": f"derived/kits/{kit}.tar", "sha256": sha, "license": "CC0 1.0", "mode": "fetch", "url": None,
              "converted_from": f"shootouts/kits/{kit}.yaml (sha256 000000000000) by lofi_rl.convert.kit 1",
              "attribution": None}
    head = f"# {kit}: sha256 {sha} ({tar.stat().st_size:,} bytes), synthetic\n# sources goes into the palette's sources\n"
    (folder / f"{kit}.yaml").write_text(head + yaml.safe_dump({"sources": {kit: source}}, sort_keys=False)
                                        + yaml.safe_dump({"choke": CHOKE, "lanes": lanes}, sort_keys=False))
    return sha


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


TARGET_LU = {"sax": 0.0, "drums": -1.5, "bass": -3.0, "keys": -6.0}      # base.yaml's mix.bus_target_lu
S_STAR = -24.0


def synthetic_levels(index_path: Path, base: Path, out: Path) -> Path:
    """A levels file shaped like the level tool's from a measuring index: GU instruments t = 0, others made up, and a
    clip fader for every variant on every reference loop and every loop's sour copy, a few thousandths of an LU off
    its target."""
    index = json.loads(index_path.read_text())
    loops = sorted(yaml.safe_load((ROOT / "shootouts" / "loops" / "briefs.yaml").read_text())["loops"])
    cases = loops + [f"{x}.sour" for x in loops]
    variants = {}
    for i, (vid, v) in enumerate(sorted(index["variants"].items())):
        gu = v["source"] == "gu"
        target = S_STAR + TARGET_LU[v["bus"]]
        clips = {}
        for j, case in enumerate(cases):
            L, off = round(-30.0 - 0.25 * i + 0.5 * j, 2), round(0.001 * ((i + j) % 5 - 2), 3)
            clips[case] = {"L": L, "F": round(target - L, 2), "lufs": target + off, "target_lufs": target,
                           "off_lu": off, "lu": target + off - S_STAR, "tried": 1}
        variants[vid] = {"bus": v["bus"], "source": v["source"], "source_sha256": v["source_sha256"],
                         "kit_map_sha256": v["kit_map_sha256"], "t": 0.0 if gu else round(-7.5 + 0.37 * i, 2),
                         "F": round(1.25 + 0.5 * i, 2), "T": TARGET_LU[v["bus"]], "clips": clips}
    out.write_text(json.dumps({"schema": "lofi-rl/shootout-levels/1", "stage": "day1",
                               "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(), "S_star": S_STAR,
                               "tolerances": {"median_tol_lu": 0.05, "clip_tol_lu": 0.1},
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


# ------------------------------------------------------------------------------------- fake renders
def fake_flac(path: Path, frames: int, *, sr: int = 44100, channels: int = 2, salt: str = "") -> Path:
    """`fLaC` + one STREAMINFO block (the last metadata block) stating `frames` samples; the MD5 field holds
    a hash of `salt`, so each fake file has its own sha256."""
    x = (sr << 44) | ((channels - 1) << 41) | (15 << 36) | frames          # 16-bit
    body = (4096).to_bytes(2, "big") * 2 + bytes(6) + x.to_bytes(8, "big") + hashlib.md5(salt.encode()).digest()
    assert len(body) == 34
    path.write_bytes(b"fLaC" + bytes([0x80]) + len(body).to_bytes(3, "big") + body)
    return path


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fake_level(doc: dict, entry: dict) -> dict | None:
    """The level record a real render of a level-matched clip carries (lofi_rl.shootout.render.with_level), made up to
    agree with fake_render's buses (-20 LUFS before a 0 dB fader); None when the clip's palette names no `level`."""
    want = ((doc.get("palettes") or {}).get(entry["palette"]) or {}).get("level")
    if not want:
        return None
    return {**want, "L": -20.0, "F": 0.0, "lufs": -20.0, "lu": TARGET_LU[want["bus"]], "target_lufs": -20.0,
            "off_lu": 0.0, "tol_lu": 0.1, "render_lufs": -20.0, "agree_lu": 0.0, "ok": True}


def fake_render(renders_dir: Path, rid: str, entry: dict, *, status: str = "ok", n: int | None = None,
                pcm: str | None = None, clip: str | None = None, level: dict | None = None) -> dict:
    d = Path(renders_dir) / rid
    d.mkdir(parents=True, exist_ok=True)
    n_ok = loop_samples(entry["brief"]["bpm"])
    pcm = pcm or _sha(f"pcm|{rid}")
    rec = {k: None for k in RECORD_KEYS}
    rec.update({
        "schema": RECORD_SCHEMA, "render": rid, "status": status, "error": None if status == "ok" else "fake failure",
        "loop": entry["loop"], "brief": entry["brief"], "stage": entry["stage"], "option": entry["option"],
        "what": entry["what"], "bus": entry["bus"], "variant": entry["variant"],
        "palette": {"key": entry["palette"], "id": entry["palette"], "sha256": _sha(entry["palette"]),
                    "label": f"{entry['palette']}@{_sha(entry['palette'])[:8]}"},
        "clip": clip or "c" + _sha(f"clip|{rid}")[:16], "n": n_ok, "sr": 44100, "bpm": entry["brief"]["bpm"],
        "lufs": -14.0, "true_peak_dbtp": -1.5, "master_gain_db": 3.0, "pcm_sha256": pcm,
        "music_sha256": pcm, "canon_sha256": _sha(entry["text"]), "compiled_sha256": _sha(f"c|{rid}"),
        "gates": {"G1-G6": "pass", "G7": "pass" if status == "ok" else "fail"},
        "buses": {b: {"lufs": -20.0, "fader_db": 0.0, "trim_db": 0.0, "post_lufs": -20.0}
                  for b in ("sax", "keys", "bass", "drums")},
        "bed_lufs": None, "sax_range": [51, 72], "expect_pcm_sha256": entry.get("expect_pcm_sha256"),
        "level": level,
    })
    if status == "ok":
        flac = fake_flac(d / "render.flac", n if n is not None else n_ok, salt=rid)
        rec["flac_sha256"] = hashlib.sha256(flac.read_bytes()).hexdigest()
        rec["flac_frames"] = n if n is not None else n_ok
        (d / "manifest.json").write_text(json.dumps({"n": n_ok, "audio_sha256": pcm}) + "\n")
    (d / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def fake_renders(doc: dict, renders_dir: Path, overrides: dict | None = None) -> dict[str, dict]:
    """A fake render for every render the draw lists; `overrides[rid]` holds keyword overrides, or None to skip."""
    out = {}
    for rid, entry in doc["renders"].items():
        kw = (overrides or {}).get(rid, {})
        if kw is None:
            continue
        out[rid] = fake_render(renders_dir, rid, entry, **{"level": fake_level(doc, entry), **kw})
    return out
