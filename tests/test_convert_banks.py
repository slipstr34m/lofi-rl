"""The six shootout SFZ banks through the converter and FluidSynth 2.6.1 (docs/CONVERTER.md). Marked cluster: they
run only with -m cluster and LOFI_ASSETS_BIG naming a folder that holds the fetched banks (assets/big on the
laptop, $REMOTE_KEEP/assets on Discovery), and skip without fluidsynth. Each bank is converted into tmp_path (the
jRhodes3d conversion is non-commercial and never lands in the repo) and checked:
- FluidSynth loads it and renders with no fault line, and logs no warning but the drum channel's missing kit;
- every key of its range sounds at velocities 40, 80 and 120;
- at velocity 100 every key plays the pitch the SFZ asks for: the source sample moved by tune= and the key
  offset, within 1 cent (log-spectrum cross-correlation against the source file);
- every key lands within 15 cents of 440·2^((k−69)/12), except the keys listed in SOURCE_OUT_OF_TUNE, where the
  SFZ itself plays further off (measured from its source samples and tune= values on 2026-10-31).
Each bank prints its pitch-error distribution (run with -s to see it). A second test parses every .sfz in the six
banks' folders and checks that no opcode of a region kept at the defaults goes unexplained. Two more check the
loops the Wurlitzer's and MTG's FLACs carry: each sample header holds its file's loop, the Wurlitzer (no loop_mode)
plays it and MTG's NL presets (loop_mode=no_loop) do not, and a held Wurlitzer note keeps sounding on it."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from helpers_sf2 import flac_riff_smpl, read_sf2, zones
from lofi_rl.convert.banks import BANKS, convert_bank
from lofi_rl.convert.sfz import parse_sfz, select
from lofi_rl.render import fluid
from lofi_rl.render.wavio import read_float

pytestmark = pytest.mark.cluster
BIG = os.environ.get("LOFI_ASSETS_BIG")
SR = 44100
# Keys where the source SFZ itself (its sample at the keycenter, moved by its tune=) is more than 15 cents from
# equal temperament, so a faithful conversion is too: the VCSL Knight's own tune= values (B-1 has tune=-52 on a
# sample within 4 cents of B) and the VSCO 2 pizz samples, which the SFZ does not tune. Only fidelity is
# asserted there.
SOURCE_OUT_OF_TUNE = {
    "vcsl-upright-knight": frozenset({23, 24, 25, 26, 27, 28, 37, 38, 39, 40, 43, 44, 101, 102, 107}),
    "vsco2-contrabass-pizz": frozenset({29, 31, 32, 45, 46}),
}
_BENIGN = "No preset found on channel 9"       # the drum channel: these banks have no bank-128 kit


def _bank_or_skip(bank: str) -> Path:
    if not BIG:
        pytest.skip("LOFI_ASSETS_BIG is not set")
    b = BANKS[bank]
    if not (Path(BIG) / b.folder / b.sfz).is_file():
        pytest.skip(f"{b.folder}/{b.sfz} is not under LOFI_ASSETS_BIG={BIG}")
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth not on PATH")
    return Path(BIG)


def _midi(path: Path, notes: list[tuple[int, int]], dur: float, gap: float) -> Path:
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("program_change", program=0))
    tps = 960                                            # 120 bpm
    for i, (key, vel) in enumerate(notes):
        track.append(mido.Message("note_on", note=key, velocity=vel, time=0 if i == 0 else round(gap * tps)))
        track.append(mido.Message("note_off", note=key, velocity=0, time=round(dur * tps)))
    track.append(mido.MetaMessage("end_of_track", time=tps))
    mid.save(path)
    return path


def _log(sf2: Path, mid: Path, wav: Path) -> list[str]:
    """FluidSynth's log for the renderer's own command line (render_midi raises on faults but keeps warnings)."""
    cmd = [fluid.fluidsynth_bin(), "-q", "-n", "-i", "-R", "0", "-C", "0", "-g", f"{fluid.GAIN}", "-r", f"{SR}",
           "-o", "synth.cpu-cores=1", "-T", "wav", "-O", "float", "-F", str(wav), str(sf2), str(mid)]
    with tempfile.TemporaryDirectory() as home:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=dict(os.environ, HOME=home))
    assert p.returncode == 0
    return [ln for ln in f"{p.stdout}\n{p.stderr}".splitlines() if ln.strip()]


def _spectrum(x: np.ndarray, sr: int, pad: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    return np.fft.rfftfreq(pad * n, 1 / sr), np.abs(np.fft.rfft(x * np.hanning(n), pad * n))


def et_cents(x: np.ndarray, sr: int, ref: float) -> float:
    """Pitch against `ref` in cents: the strongest of harmonics 1-4 within ±40 cents of h·ref, read as the
    power-weighted centroid of its peak (robust to a piano's split unison partials)."""
    f, mag = _spectrum(x, sr, 8)
    best = None
    for h in (1, 2, 3, 4):
        lo, hi = h * ref * 2 ** (-40 / 1200), h * ref * 2 ** (40 / 1200)
        if hi > sr / 2 - 500:
            break
        band = np.nonzero((f >= lo) & (f <= hi))[0]
        if best is None or mag[band].max() > best[0]:
            best = (mag[band].max(), band, h)
    _, band, h = best
    m, c = mag[band], 1200 * np.log2(f[band] / (h * ref))
    sel = m >= 0.3 * m.max()
    return float(np.sum(m[sel] ** 2 * c[sel]) / np.sum(m[sel] ** 2))


def residual_cents(src: np.ndarray, src_sr: int, out: np.ndarray, f0: float, shift: float) -> float:
    """How far `out` is from `src` moved by `shift` cents: cross-correlation of log-frequency magnitude spectra
    (harmonics 1-8 of f0), searched over ±60 cents in 0.5-cent steps, refined parabolically."""
    step, span = 0.5, 60
    f_lo, f_hi = f0 * 0.7, min(f0 * 8, 0.45 * src_sr, 0.45 * SR / 2 ** (shift / 1200))
    n = int(span / step)
    grid = f_lo * 2 ** (np.arange(-n, int(1200 * np.log2(f_hi / f_lo) / step) + n) * step / 1200)

    def logspec(x, sr, g):
        f, mag = _spectrum(x, sr, 4)
        s = np.log1p(np.interp(g, f, mag) / (mag.max() + 1e-30) * 1000)
        return s - s.mean()
    a, b = logspec(src, src_sr, grid), logspec(out, SR, grid * 2 ** (shift / 1200))
    core = slice(n, len(a) - n)
    cs = [float(np.dot(a[core], b[core.start + s:core.stop + s])) for s in range(-n, n + 1)]
    k = int(np.argmax(cs))
    p = 0.0
    if 0 < k < len(cs) - 1:
        p = 0.5 * (cs[k - 1] - cs[k + 1]) / (cs[k - 1] - 2 * cs[k] + cs[k + 1])
    return (k - n + p) * step


@pytest.mark.parametrize("bank", list(BANKS))
def test_bank_converts_loads_sounds_and_plays_in_tune(bank, tmp_path):
    big = _bank_or_skip(bank)
    conv = convert_bank(bank, big)
    sf2 = tmp_path / f"{bank}.sf2"
    sf2.write_bytes(conv.data)
    lo, hi = conv.report["keys"]
    keys = list(range(lo, hi + 1))

    # loads, renders with no fault line, and logs nothing but the drum channel's missing kit
    one = _midi(tmp_path / "one.mid", [(keys[len(keys) // 2], 100)], 0.3, 0.0)
    noise = [ln for ln in _log(sf2, one, tmp_path / "one.wav") if _BENIGN not in ln]
    assert not noise, f"FluidSynth logged: {noise[:4]}"

    # every key sounds at velocities 40, 80 and 120 (render_midi raises on any fault line)
    dur, gap = 0.5, 0.1
    notes = [(k, v) for k in keys for v in (40, 80, 120)]
    y, sr = read_float(fluid.render_midi(sf2, _midi(tmp_path / "all.mid", notes, dur, gap), tmp_path / "all.wav"))
    assert sr == SR
    silent = []
    for i, (k, v) in enumerate(notes):
        a = int(round(i * (dur + gap) * SR))
        if np.sqrt(np.mean(y[:, a:a + int(dur * SR)].astype(np.float64) ** 2)) < 1e-6:
            silent.append((k, v))
    assert not silent, f"silent notes (key, velocity): {silent[:10]}"

    # pitch at velocity 100: fidelity to the SFZ at every key, and equal temperament where the SFZ is in tune
    b = BANKS[bank]
    kept, _ = select(parse_sfz(big / b.folder / b.sfz), keep=b.keep)
    pdur, pgap = 1.2, 0.1
    yp, _ = read_float(fluid.render_midi(sf2, _midi(tmp_path / "pitch.mid", [(k, 100) for k in keys], pdur, pgap),
                                         tmp_path / "pitch.wav"))
    fidelity, et, cache = {}, {}, {}
    for i, k in enumerate(keys):
        (r,) = [z for z in kept if z.lokey <= k <= z.hikey and z.lovel <= 100 <= z.hivel]
        shift = r.tune + (k - r.keycenter) * r.keytrack
        if r.sample not in cache:
            x, src_sr = sf.read(str(r.sample), dtype="float64", always_2d=True, frames=r.offset + 4 * 48000)
            cache[r.sample] = (x[r.offset:, 0], src_sr)
        x, src_sr = cache[r.sample]
        a = int(round(i * (pdur + pgap) * SR))
        out = yp[0, a + int(0.08 * SR):a + int(0.9 * SR)].astype(np.float64)
        ratio = 2 ** (shift / 1200)                       # the render's 0.08-0.9 s window reads this much source
        src = x[int(0.08 * ratio * src_sr):int(0.9 * ratio * src_sr)]
        fidelity[k] = residual_cents(src, src_sr, out, 440 * 2 ** ((r.keycenter - 69) / 12), shift)
        et[k] = et_cents(out, SR, 440 * 2 ** ((k - 69) / 12))
    e = np.array(list(et.values()))
    exempt = SOURCE_OUT_OF_TUNE.get(bank, frozenset())
    print(f"\n{bank}: {len(keys)} keys {lo}-{hi}; vs equal temperament: min {e.min():+.1f}, median "
          f"{np.median(e):+.1f}, max {e.max():+.1f}, max |error| {np.abs(e).max():.1f} cents; keys over 15: "
          f"{[k for k in keys if abs(et[k]) > 15]}; vs the SFZ's own pitch: max |error| "
          f"{max(abs(v) for v in fidelity.values()):.2f} cents")
    assert max(abs(v) for v in fidelity.values()) < 1.0, {k: round(v, 2) for k, v in fidelity.items() if abs(v) >= 1}
    off = {k: round(et[k], 1) for k in keys if abs(et[k]) > 15 and k not in exempt}
    assert not off, f"keys more than 15 cents from equal temperament: {off}"


_FOLDERS = sorted({b.folder for b in BANKS.values()})


@pytest.mark.parametrize("folder", _FOLDERS)
def test_every_program_in_the_banks_parses_and_every_kept_opcode_is_accounted_for(folder):
    """Every .sfz in the six banks' folders (the 23 programs and maps, chosen or not) parses and evaluates at the
    defaults, and every opcode of a region kept there is either mapped or dropped for a stated reason (none
    falls through to the generic one). Samples need not be fetched: only the chosen programs' were."""
    if not BIG or not (Path(BIG) / folder).is_dir():
        pytest.skip(f"{folder} is not under LOFI_ASSETS_BIG")
    files = sorted((Path(BIG) / folder).rglob("*.sfz"))
    assert files
    for f in files:
        kept, _ = select(parse_sfz(f), need_samples=False)
        generic = sorted({k for r in kept for k, why in r.unexpressed.items() if why == "no SF2 equivalent"})
        assert not generic, f"{f.name}: opcodes with no stated reason: {generic}"


# The FLAC banks whose samples carry a smpl loop in 'riff' APPLICATION blocks (flac --keep-foreign-metadata), and
# the sampleModes their SFZs ask for: the Wurlitzer sets no loop_mode, so its samples' own loops play (1); MTG's
# NL ("non-looped") presets set loop_mode=no_loop (0).
FLAC_LOOPED = {"gsullivan-wurlitzer": 1, "mtg-tenor": 0, "mtg-alto": 0}


@pytest.mark.parametrize("bank", list(FLAC_LOOPED))
def test_flac_banks_keep_their_embedded_loops_and_play_them_as_the_sfz_says(bank):
    big = _bank_or_skip(bank)
    b = BANKS[bank]
    kept, _ = select(parse_sfz(big / b.folder / b.sfz), keep=b.keep)
    by_range = {(r.lokey, r.hikey, r.lovel, r.hivel): r for r in kept}
    assert len(by_range) == len(kept)
    sf2 = read_sf2(convert_bank(bank, big).data)
    zs = zones(sf2)
    assert len(zs) == len(kept)
    modes, wrong = set(), []
    for z in zs:
        g = dict(z["gens"])
        r = by_range[(g[43] & 0xFF, g[43] >> 8, g[44] & 0xFF, g[44] >> 8)]
        unity, loops = flac_riff_smpl(r.sample)
        ltype, start, end = loops[0]
        rec = sf2["shdr"][g[53]]
        if (ltype, rec[3] - rec[1], rec[4] - rec[1]) != (0, start, end):
            wrong.append((r.sample.name, (rec[3] - rec[1], rec[4] - rec[1]), (start, end)))
        modes.add(g.get(54, 0))
    assert not wrong, f"sample headers without their file's loop: {wrong[:4]}"
    assert modes == {FLAC_LOOPED[bank]}


def test_a_held_wurlitzer_note_keeps_sounding_on_its_loop_through_the_sfz_hold(tmp_path):
    # key 74 at velocity 20 plays db5pp.flac (keycenter 73, tune=-5): 15,788 frames, loop 13607-15747. The SFZ holds
    # it at full level for 5 x (1 - 74/127) = 2.09 s, while its points run out after 0.34 s at this key. Held 2 s,
    # one loop period at 1.5 s must match the loop's first pass (both inside the hold) within 1 dB.
    bank = "gsullivan-wurlitzer"
    big = _bank_or_skip(bank)
    b = BANKS[bank]
    kept, _ = select(parse_sfz(big / b.folder / b.sfz), keep=b.keep)
    (r,) = [z for z in kept if z.lokey <= 74 <= z.hikey and z.lovel <= 20 <= z.hivel]
    assert r.sample.name == "db5pp.flac"
    _, loops = flac_riff_smpl(r.sample)
    _, start, end = loops[0]
    step = 2 ** ((r.tune + (74 - r.keycenter) * r.keytrack) / 1200) * sf.info(str(r.sample)).samplerate / SR
    first, period = start / step / SR, (end - start) / step / SR       # source points per output sample: step
    hold = r.env["hold"].at(20, 74)
    assert first + period < 1.5 < 1.5 + period < min(hold, 2.0)
    path = tmp_path / f"{bank}.sf2"
    path.write_bytes(convert_bank(bank, big).data)
    y, _ = read_float(fluid.render_midi(path, _midi(tmp_path / "held.mid", [(74, 20)], 2.0, 0.0), tmp_path / "h.wav"))

    def db(t: float) -> float:
        seg = y[:, int(round(t * SR)):int(round((t + period) * SR))].astype(np.float64)
        return 10 * math.log10(np.mean(seg ** 2) + 1e-30)
    assert db(1.5) == pytest.approx(db(first), abs=1.0), f"{db(1.5):.1f} dB at 1.5 s, {db(first):.1f} on the first pass"
