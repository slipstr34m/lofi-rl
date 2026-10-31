"""Independent readers and writers for the converter's tests (docs/CONVERTER.md), written from the file formats'
byte layouts rather than from lofi_rl.convert's code:
- read_sf2: a SoundFont 2 file's chunks, INFO, sample points and pdta records;
- write_flac: a FLAC file carrying a WAV's chunks the way `flac --keep-foreign-metadata` stores them;
- flac_riff_smpl: the smpl chunk inside those blocks.
Not a test module."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import soundfile as sf

__all__ = ["flac_riff_smpl", "name", "read_sf2", "signed", "write_flac", "zones"]


# --------------------------------------------------------------- an independent SoundFont reader
def read_sf2(data: bytes) -> dict:
    """Parse a SoundFont 2 file straight from the spec's byte layout (not lofi_rl.convert.sf2's code)."""
    assert data[:4] == b"RIFF" and data[8:12] == b"sfbk"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8, "RIFF size covers the whole file"

    def walk(start: int, end: int) -> list[tuple[bytes, int, int]]:
        out, pos = [], start
        while pos < end:
            cid, size = struct.unpack_from("<4sI", data, pos)
            out.append((cid, pos + 8, size))
            pos += 8 + size + (size & 1)
        assert pos == end, "sub-chunks tile their parent exactly"
        return out
    lists = {}
    for cid, off, size in walk(12, len(data)):
        assert cid == b"LIST"
        lists[data[off:off + 4]] = {c: (o, s) for c, o, s in walk(off + 4, off + size)}
        lists.setdefault(b"_order", []).append(data[off:off + 4])
    sf2 = {"order": lists[b"_order"], "info": {}, "chunks": {k: list(v) for k, v in lists.items() if k != b"_order"}}
    for cid, (off, size) in lists[b"INFO"].items():
        sf2["info"][cid.decode()] = data[off:off + size]
    smpl_off, smpl_size = lists[b"sdta"][b"smpl"]
    sf2["smpl"] = np.frombuffer(data, "<i2", smpl_size // 2, smpl_off)
    if b"sm24" in lists[b"sdta"]:
        o, s = lists[b"sdta"][b"sm24"]
        sf2["sm24"] = np.frombuffer(data, np.uint8, s, o)
    pd = lists[b"pdta"]
    assert list(pd) == [b"phdr", b"pbag", b"pmod", b"pgen", b"inst", b"ibag", b"imod", b"igen", b"shdr"]

    def records(cid: bytes, fmt: str) -> list[tuple]:
        o, s = pd[cid]
        n = struct.calcsize(fmt)
        assert s % n == 0, f"{cid} size {s} is not a multiple of {n}"
        return [struct.unpack_from(fmt, data, o + i * n) for i in range(s // n)]
    sf2["phdr"] = records(b"phdr", "<20sHHHIII")
    sf2["pbag"] = records(b"pbag", "<HH")
    sf2["pmod"] = records(b"pmod", "<HHhHH")
    sf2["pgen"] = records(b"pgen", "<HH")
    sf2["inst"] = records(b"inst", "<20sH")
    sf2["ibag"] = records(b"ibag", "<HH")
    sf2["imod"] = records(b"imod", "<HHhHH")
    sf2["igen"] = records(b"igen", "<HH")
    sf2["shdr"] = records(b"shdr", "<20sIIIIIBbHH")
    return sf2


def zones(sf2: dict) -> list[dict]:
    """The instrument zones of instrument 0: generator (id, raw word) pairs in file order, and modulators."""
    start, end = sf2["inst"][0][1], sf2["inst"][1][1]
    out = []
    for b in range(start, end):
        g0, m0 = sf2["ibag"][b]
        g1, m1 = sf2["ibag"][b + 1]
        out.append({"gens": sf2["igen"][g0:g1], "mods": sf2["imod"][m0:m1]})
    return out


def signed(word: int) -> int:
    return word - 65536 if word >= 32768 else word


def name(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii")


# ------------------------------------------------------------------- FLAC with foreign metadata
def _flac_blocks(data: bytes) -> tuple[list[tuple[int, bytes]], int]:
    """(block type, body) of every metadata block, and the offset where the audio frames start."""
    assert data[:4] == b"fLaC"
    pos, out = 4, []
    while True:
        head = data[pos]
        size = int.from_bytes(data[pos + 1:pos + 4], "big")
        out.append((head & 0x7F, data[pos + 4:pos + 4 + size]))
        pos += 4 + size
        if head & 0x80:
            return out, pos


def write_flac(path: Path, frames: np.ndarray, rate: int, bits: int, loop: tuple[int, int] | None = None,
               unity: int = 60) -> Path:
    """A FLAC file of integer frames (n,) or (n, channels) at `bits`, carrying its WAV's chunks the way
    `flac --keep-foreign-metadata` does: one APPLICATION block (type 2) with id 'riff' per chunk, in the WAV's
    order (the RIFF header, fmt, the data chunk's header without its points, then smpl when `loop` = (start, end
    exclusive) is given). libsndfile encodes the points; the blocks are spliced in after its own."""
    x = np.asarray(frames, dtype=np.int64)
    x = x[:, None] if x.ndim == 1 else x
    ch = x.shape[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (x << (32 - bits)).astype(np.int32), rate, subtype=f"PCM_{bits}", format="FLAC")
    blocks, audio = _flac_blocks(path.read_bytes())
    data_bytes = x.shape[0] * ch * bits // 8
    chunks = [b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, rate, rate * ch * bits // 8, ch * bits // 8, bits),
              b"data" + struct.pack("<I", data_bytes)]
    if loop is not None:
        body = struct.pack("<9I", 0, 0, 1_000_000_000 // rate, unity, 0, 0, 0, 1, 0)
        body += struct.pack("<6I", 0, 0, loop[0], loop[1] - 1, 0, 0)       # dwEnd is the loop's last point
        chunks.append(b"smpl" + struct.pack("<I", len(body)) + body)
    size = 4 + sum(len(c) for c in chunks) + data_bytes + (data_bytes & 1)
    blocks += [(2, b"riff" + chunk) for chunk in [b"RIFF" + struct.pack("<I", size) + b"WAVE", *chunks]]
    out = [b"fLaC"]
    for i, (btype, body) in enumerate(blocks):
        out.append(bytes([(0x80 if i == len(blocks) - 1 else 0) | btype]) + len(body).to_bytes(3, "big") + body)
    data = path.read_bytes()
    path.write_bytes(b"".join(out) + data[audio:])
    return path


def flac_riff_smpl(path: Path) -> tuple[int, list[tuple[int, int, int]]] | None:
    """The smpl chunk carried in a FLAC's 'riff' APPLICATION blocks: (unity note, [(type, start, end exclusive)]),
    or None when there is none."""
    blocks, _ = _flac_blocks(Path(path).read_bytes())
    for btype, body in blocks:
        if btype == 2 and body[:8] == b"riffsmpl":
            size = struct.unpack_from("<I", body, 8)[0]
            chunk = body[12:12 + size]
            unity, count = struct.unpack_from("<I", chunk, 12)[0], struct.unpack_from("<I", chunk, 28)[0]
            loops = [struct.unpack_from("<6I", chunk, 36 + 24 * i) for i in range(count)]
            return unity, [(t, s, e + 1) for _cue, t, s, e, _f, _p in loops]
    return None
