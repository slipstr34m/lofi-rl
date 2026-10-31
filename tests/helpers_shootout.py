"""Fake shootout renders for the batch builder's tests (not a test module).

A fake render folder holds what `lofi_rl.shootout.render` writes and the builder reads: `record.json` (the same
keys as a real record, `tests/test_shootout_e2e.py` compares them), `manifest.json` (n and audio_sha256) and
`render.flac`, a FLAC STREAMINFO header with no audio frames. The rating page's checks read only that header and
the file's sha256, so the batches built from these pass `python -m lofi_rl.ui.manifest check` with no audio
encoded. Never music, never committed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lofi_rl.compile.timeline import loop_samples
from lofi_rl.shootout.render import RECORD_KEYS, RECORD_SCHEMA


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


def fake_render(renders_dir: Path, rid: str, entry: dict, *, status: str = "ok", n: int | None = None,
                pcm: str | None = None, clip: str | None = None) -> dict:
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
        out[rid] = fake_render(renders_dir, rid, entry, **kw)
    return out
