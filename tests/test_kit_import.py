"""The one-shot kit importer (docs/SHOOTOUTS.md §7.4; CD-30, CD-33, CD-35).

`python -m lofi_rl.convert.kit <spec.yaml> --big <assets/big> --out <kit.tar>` reads a kit spec (shootouts/kits/),
builds every one-shot it names (each mic's file, stereo as (L + R)/2, the mics summed with their dB gains in their
recorded timing, 48 kHz resampled once, the lead-in trimmed at the first sample within -30 dB of the peak, -20 dB on
a shaker lane, keeping 1 ms with a raised-cosine fade-in, trailing digital silence dropped), writes them as a
byte-deterministic tar of 44.1 kHz mono float32 WAVs and prints the palette's source entry, `choke:` list and
`lanes:` block. A borrowed lane's level copies the reference kit's shaker-to-snare difference on a fixed grid.

Synthetic kits in tmp_path cover the default suite (the GU reference renders through FluidSynth only in
test_the_reference_comes_from_the_dev_kit). The real banks are imported by the cluster-marked tests, which skip unless
LOFI_ASSETS_BIG (or LOFI_ASSETS) names the assets/big folder; run them with -s to see each kit's report."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from helpers_compile import DEV, ROOT
from helpers_render import needs_fluidsynth
from lofi_rl.convert import kit as K
from lofi_rl.convert.resample import to_44k
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import LAYERS, load_palette
from lofi_rl.render import sampler
from lofi_rl.render.sampler import kit_shots

DEV_DOC = yaml.safe_load(DEV.read_text())
DEV_CHOKE = [list(p) for p in DEV_DOC["kits"]["jazz"]["choke"]]
Q = 2.0 ** -12                     # synthetic sample values are multiples of Q, so every sum and halving is exact


# ----------------------------------------------------------------------------------------- synthetic kits
def shot_signal(seed: int, lead: int, body: int = 1500, tail_zeros: int = 300, level: float = 0.5,
                floor: float = 0.0) -> np.ndarray:
    """A decaying noise burst after `lead` samples of noise at `floor` (relative to `level`), then digital silence."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-1, 1, lead) * floor * level,
                        rng.uniform(-1, 1, body) * level * np.exp(-np.arange(body) / (body / 4)),
                        np.zeros(tail_zeros)])
    x[lead] = level                                   # the peak sits on the first sample of the burst
    return np.round(x / Q) * Q


def write(path: Path, x: np.ndarray, sr: int, subtype: str = "FLOAT") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype=subtype)


SPEC = """\
id: kit_synth
license: CC0 1.0
attribution: null
mics: {a: 0.0, b: -6.0}
lanes:
  kick:
    path: synth/kick/{mic}_vl{vl}_rr{rr}.wav
    vl: {soft: 1, med: 2, hard: 3}
    rr: [1, 2]
  snare:
    path: synth/snare/{mic}_vl{vl}.wav
    vl: {soft: 2, med: 3, hard: 4}
    rr: neighbours
  rim:
    mics: {c: 0.0}
    path: synth/rim/{mic}_vl{vl}_rr{rr}.wav
    vl: {soft: 1, med: 2, hard: 3}
    rr: [1]
  hat:
    mics: {c: 0.0}
    path: synth/hat/{mic}_vl{vl}_rr{rr}.wav
    vl: {soft: 1, med: 1, hard: 2}
    rr: [1, 2]
  ohat:
    mics: {c: 0.0}
    path: synth/ohat/{mic}_vl{vl}_rr{rr}.wav
    vl: {soft: 1, med: 2, hard: 3}
    rr: [1]
  shaker:
    borrowed: BORROWED
    mics: {close: {db: 0.0, label: Close}, oh: {db: 0.0, label: Overhead}}
    path: synth/shaker/{mic}/Shake1{dir}_rr{rr}_{label}.wav
    all_layers: [{dir: D, rr: 2}, {dir: U, rr: 1}, {dir: D, rr: 3}, {dir: U, rr: 4}]
"""


def synth_kit(tmp_path: Path, borrowed: bool = False) -> tuple[Path, Path]:
    """A synthetic sample folder (48 kHz kick and snare on mics a (mono) and b (stereo); 44.1 kHz rim, hats and a
    two-mic shaker) and its spec. Returns (big, spec path)."""
    big = tmp_path / "big"
    seed = iter(range(10_000))
    for mic, ch in (("a", 1), ("b", 2)):
        for vl in (1, 2, 3):
            for rr in (1, 2):
                x = shot_signal(next(seed), lead=600 + 7 * vl + rr, floor=0.01)
                write(big / f"synth/kick/{mic}_vl{vl}_rr{rr}.wav", np.stack([x, x * 0.5], 1) if ch == 2 else x, 48000)
        for vl in range(1, 6):
            x = shot_signal(next(seed), lead=300 + 11 * vl)
            write(big / f"synth/snare/{mic}_vl{vl}.wav", np.stack([x, -x * 0.25], 1) if ch == 2 else x, 48000)
    for lane, vls, rrs in (("rim", (1, 2, 3), (1,)), ("hat", (1, 2), (1, 2)), ("ohat", (1, 2, 3), (1,))):
        for vl in vls:
            for rr in rrs:
                write(big / f"synth/{lane}/c_vl{vl}_rr{rr}.wav", shot_signal(next(seed), lead=20 + 50 * vl), 44100)
    for d, rr in (("D", 2), ("U", 1), ("D", 3), ("U", 4)):
        x = shot_signal(next(seed), lead=900, floor=0.05, level=0.3)
        write(big / f"synth/shaker/close/Shake1{d}_rr{rr}_Close.wav", x, 44100)
        write(big / f"synth/shaker/oh/Shake1{d}_rr{rr}_Overhead.wav", np.stack([x * 0.5, x * 0.25], 1), 44100)
    spec = tmp_path / "kit_synth.yaml"
    spec.write_text(SPEC.replace("BORROWED", "true" if borrowed else "false"))
    return big, spec


def synth_reference(level_db: float = -3.0) -> K.Reference:
    """A reference kit of synthetic one-shots: its shaker sits `level_db` under its snare (roughly)."""
    snare = shot_signal(1, lead=0, body=6000, level=0.8)
    shaker = shot_signal(2, lead=0, body=3000, level=0.8 * 10 ** (level_db / 20))
    return K.Reference("synthetic", tuple(tuple(p) for p in DEV_CHOKE), (shaker,), (snare,), 0.71)


# ----------------------------------------------------------------------------------- the pipeline pieces
def test_the_layer_rule_gives_the_plans_picks():
    # CD-33: soft / med / hard take layers floor(0.25 N + 1/2), floor(0.6 N + 1/2) and floor(0.9 N + 1/2).
    table = {4: (1, 2, 4), 36: (9, 22, 32), 16: (4, 10, 14), 22: (6, 13, 20), 12: (3, 7, 11), 10: (3, 6, 9),
             14: (4, 8, 13)}
    for n, picks in table.items():
        assert tuple(K.cd33_layers(n)[layer] for layer in LAYERS) == picks, n


def test_stereo_becomes_the_mean_of_its_channels(tmp_path):
    left, right = shot_signal(5, lead=200), np.roll(shot_signal(6, lead=200), 30) * 0.5
    write(tmp_path / "st.wav", np.stack([left, right], 1), 48000)
    write(tmp_path / "mono.wav", (left + right) / 2, 48000)
    m, sr = K.read_mono(tmp_path / "st.wav")
    assert sr == 48000 and np.array_equal(m, (left + right) * 0.5)
    a, _ = K.one_shot([(tmp_path / "st.wav", 0.0)], "snare")
    b, _ = K.one_shot([(tmp_path / "mono.wav", 0.0)], "snare")
    assert a.dtype == np.float32 and np.array_equal(a, b)


def test_two_mics_keep_their_recorded_timing(tmp_path):
    # Impulses 100 samples apart at 48 kHz, one per mic, come out 100 x 147/160 = 91.9 samples apart.
    for mic, at, height in (("a", 1000, 0.5), ("b", 1100, 0.3)):
        x = np.zeros(4000)
        x[at] = height
        write(tmp_path / f"{mic}.wav", x, 48000)
    y, info = K.one_shot([(tmp_path / "a.wav", 0.0), (tmp_path / "b.wav", 0.0)], "kick")
    first = int(np.argmax(np.abs(y)))                                 # the taller pulse, mic a's
    second = first + 40 + int(np.argmax(np.abs(y[first + 40:])))      # past the first pulse's ringing
    assert abs((second - first) - 92) <= 1, (first, second)
    assert info["rate_in"] == 48000 and info["resampled"]


def test_the_mics_are_summed_with_their_gains():
    a, b = shot_signal(8, lead=400), shot_signal(9, lead=410)
    mixed, sr = K.premix([(a, 44100, 0.0), (b[:1000], 44100, -6.0)])
    want = a.copy()
    want[:1000] += b[:1000] * 10.0 ** (-6.0 / 20.0)
    assert sr == 44100 and np.array_equal(mixed, want)
    with pytest.raises(K.KitError, match="44100 and 48000 Hz"):
        K.premix([(a, 44100, 0.0), (b, 48000, 0.0)])


def test_the_lead_in_is_trimmed_to_1_ms_before_the_threshold_with_a_raised_cosine(tmp_path):
    x = shot_signal(11, lead=700, floor=0.02).astype(np.float32)
    peak = float(np.abs(x).max())
    i0 = int(np.flatnonzero(np.abs(x) >= peak * 10 ** (-30 / 20))[0])
    y, info = K.trim(x, -30.0)
    assert (info["crossing"], info["trimmed"], info["padded"]) == (i0, i0 - K.PRE, 0)
    assert K.PRE == 44 and y.dtype == np.float32
    assert y[K.PRE] == x[i0] and y[0] == 0.0
    w = np.array([0.5 * (1 - np.cos(np.pi * k / K.PRE)) for k in range(K.PRE)])
    assert np.allclose(y[:K.PRE], x[i0 - K.PRE:i0] * w, atol=1e-9)
    assert y[-1] != 0 and len(y) == int(np.flatnonzero(x)[-1]) - (i0 - K.PRE) + 1     # trailing zeros dropped
    assert int(np.flatnonzero(np.abs(y) >= peak * 10 ** (-30 / 20))[0]) == K.PRE
    # A burst that starts within 1 ms of the file's start is padded with zeros, so it too sits at sample 44.
    early = shot_signal(12, lead=10).astype(np.float32)
    y, info = K.trim(early, -30.0)
    assert (info["crossing"], info["trimmed"], info["padded"]) == (10, 0, 34)
    assert int(np.flatnonzero(np.abs(y) >= np.abs(y).max() * 10 ** (-1.5))[0]) == K.PRE
    with pytest.raises(K.KitError, match="silent"):
        K.trim(np.zeros(100, np.float32), -30.0)


def test_a_shaker_lane_trims_at_minus_20_db(tmp_path):
    x = shot_signal(13, lead=600).astype(np.float32)
    x[300:600] = np.float32(0.5 * 10 ** (-25 / 20))     # a build-up between -30 and -20 dB of the peak
    write(tmp_path / "s.wav", x, 44100)
    assert K.TRIM_DB == {"shaker": -20.0} and K.DEFAULT_TRIM_DB == -30.0
    _, shaker = K.one_shot([(tmp_path / "s.wav", 0.0)], "shaker")
    _, kick = K.one_shot([(tmp_path / "s.wav", 0.0)], "kick")
    assert (shaker["crossing"], kick["crossing"]) == (600, 300)


def test_a_borrowed_lane_reproduces_the_reference_shaker_to_snare_difference(tmp_path):
    ref = synth_reference(-3.0)
    kit_snare = (shot_signal(21, lead=0, body=5000, level=0.9), shot_signal(22, lead=0, body=5000, level=0.85))
    kit_shaker = (shot_signal(23, lead=0, body=2000, level=0.2), shot_signal(24, lead=0, body=2500, level=0.25))
    offset, terms = K.borrowed_offset(kit_shaker, kit_snare, ref)
    assert offset == round(offset, 2) and terms["offset_db"] == offset
    want = K.grid_lufs(ref.shaker, ref.gain) - K.grid_lufs(ref.snare, ref.gain)
    got = K.grid_lufs(kit_shaker, ref.gain, db=offset) - K.grid_lufs(kit_snare, ref.gain)
    assert abs(got - want) < 0.1 and abs(got - want) <= 0.006
    # The grid: x on every 8th note for 2 bars at 85 BPM, round robin as the compiler deals it.
    n, hits = K.grid_hits("shaker", 2, 0.71)
    assert n == 249_040 and [h.on for h in hits] == [k * 15_565 for k in range(16)]
    assert [h.rr for h in hits] == [h % 2 for h in range(16)] and {h.layer for h in hits} == {"med"}


# --------------------------------------------------------------------------------------------- end to end
def test_the_archive_is_byte_deterministic_and_holds_44k_mono_float32_one_shots(tmp_path):
    big, spec = synth_kit(tmp_path)
    ref = synth_reference()
    a = K.import_kit(K.load_spec(spec), big, reference=ref)
    b = K.import_kit(K.load_spec(spec), big, reference=ref)
    assert a.data == b.data and a.sha256 == hashlib.sha256(a.data).hexdigest()
    assert not zipfile.is_zipfile(io.BytesIO(a.data))
    with tarfile.open(fileobj=io.BytesIO(a.data)) as t:
        members = t.getmembers()
        names = [m.name for m in members]
        assert names == sorted(names) and len(names) == len(set(names)) == len(a.members)
        for m in members:
            assert m.isreg() and (m.mtime, m.uid, m.gid, m.mode, m.uname, m.gname) == (0, 0, 0, 0o644, "", "")
            assert m.pax_headers == {}
            data = t.extractfile(m).read()
            info = sf.info(io.BytesIO(data))
            assert (info.samplerate, info.channels, info.subtype) == (44100, 1, "FLOAT"), m.name
            x, _ = sf.read(io.BytesIO(data), dtype="float32")
            assert np.array_equal(x, a.members[m.name])
            thr = np.abs(x).max() * 10 ** ((-20 if m.name.startswith("shaker/") else -30) / 20)
            assert abs(int(np.flatnonzero(np.abs(x) >= thr)[0]) - 44) <= 1, m.name
            assert x[-1] != 0, m.name
    assert a.data[257:263] == b"ustar\x00" and b"PaxHeader" not in a.data
    # The kick (48 kHz) went through the resampler; the rim (44.1 kHz) did not.
    assert a.report["shots"]["kick/vl1_rr1.wav"]["resampled"]
    assert not a.report["shots"]["rim/vl1_rr1.wav"]["resampled"]


def test_the_lanes_layers_and_round_robins_follow_the_spec(tmp_path):
    big, spec = synth_kit(tmp_path)
    imp = K.import_kit(K.load_spec(spec), big, reference=synth_reference())
    assert imp.lanes["kick"] == {layer: [f"kick/vl{vl}_rr{rr}.wav" for rr in (1, 2)]
                                 for layer, vl in zip(LAYERS, (1, 2, 3))}
    assert imp.lanes["snare"]["med"] == ["snare/vl2.wav", "snare/vl3.wav", "snare/vl4.wav"]
    assert imp.lanes["hat"]["soft"] == imp.lanes["hat"]["med"] == ["hat/vl1_rr1.wav", "hat/vl1_rr2.wav"]
    shaker = ["shaker/D_rr2.wav", "shaker/U_rr1.wav", "shaker/D_rr3.wav", "shaker/U_rr4.wav"]
    assert all(imp.lanes["shaker"][layer] == shaker for layer in LAYERS)
    files = imp.report["shots"]["kick/vl2_rr1.wav"]["files"]
    assert [(f["path"], f["db"]) for f in files] == [("synth/kick/a_vl2_rr1.wav", 0.0),
                                                      ("synth/kick/b_vl2_rr1.wav", -6.0)]
    assert len(imp.members) == 6 + 5 + 3 + 4 + 3 + 4      # a member per distinct one-shot, shared across layers


def test_a_palette_with_the_printed_blocks_loads_with_verify(tmp_path, capsys):
    big, spec = synth_kit(tmp_path, borrowed=True)
    out = big / "derived" / "kits" / "kit_synth.tar"
    ref = synth_reference()
    imp = K.import_kit(K.load_spec(spec), big, reference=ref)
    out.parent.mkdir(parents=True)
    out.write_bytes(imp.data)
    printed = K.blocks_text(imp, imp.blocks(K.source_path(out, big)))
    blocks = yaml.safe_load(printed)
    assert printed.startswith("# kit_synth: sha256 " + imp.sha256)
    assert blocks["choke"] == DEV_CHOKE
    assert blocks["sources"]["kit_synth"]["path"] == "derived/kits/kit_synth.tar"
    assert blocks["lanes"]["shaker"]["db"] == imp.offsets["shaker"] != 0.0
    assert all(blocks["lanes"][lane]["db"] == 0.0 and blocks["lanes"][lane]["pan"] == 0.0 for lane in LANES[:5])
    doc = copy.deepcopy(DEV_DOC)
    doc["assets_root"] = str(big)
    doc["sources"].update(blocks["sources"])
    doc["kits"]["jazz"] = {"bass": doc["kits"]["jazz"]["bass"], "choke": blocks["choke"], "lanes": blocks["lanes"]}
    path = tmp_path / "pal.yaml"
    path.write_text(yaml.safe_dump(doc))
    pal = load_palette(path, verify=True)
    kit = pal.kits["jazz"]
    assert all(kit.lanes[lane].layers[layer] for lane in LANES for layer in LAYERS)
    shots = kit_shots(pal, "jazz")
    for lane in LANES:
        for layer in LAYERS:
            got = shots.shots[(lane, layer)]
            want = [imp.members[m].astype(np.float64) for m in imp.lanes[lane][layer]]
            assert len(got) == len(want) and all(np.array_equal(g, w) for g, w in zip(got, want)), (lane, layer)
    assert shots.lanes["shaker"] == (0.0, imp.offsets["shaker"])


def test_the_cli_gives_the_same_archive_in_a_fresh_process(tmp_path):
    big, spec = synth_kit(tmp_path)
    one, two = tmp_path / "one.tar", tmp_path / "two.tar"
    assert K.main([str(spec), "--big", str(big), "--out", str(one), "--json", str(tmp_path / "r.json")]) == 0
    env = dict(os.environ, PYTHONHASHSEED="4242", PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, "-m", "lofi_rl.convert.kit", str(spec), "--big", str(big), "--out",
                           str(two)], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert one.read_bytes() == two.read_bytes()
    blocks = yaml.safe_load(proc.stdout)
    assert blocks["sources"]["kit_synth"]["sha256"] == hashlib.sha256(one.read_bytes()).hexdigest()
    assert blocks["choke"] == DEV_CHOKE
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["sha256"] == blocks["sources"]["kit_synth"]["sha256"] and report["members"] == 25
    stats = report["lead_in"]
    trims = [s["trimmed"] for s in report["shots"].values()]
    assert (stats["min"], stats["median"], stats["max"]) == (min(trims), statistics.median(trims), max(trims))


@needs_fluidsynth
def test_the_reference_comes_from_the_dev_kit(tmp_path):
    ref = K.reference_from_palette(DEV, "jazz", cache=tmp_path / "cache")
    assert ref.name == "dev_gu.jazz" and [list(p) for p in ref.choke] == DEV_CHOKE and ref.gain == 0.71
    pal = load_palette(DEV, verify=False)
    for lane, got in (("shaker", ref.shaker), ("snare", ref.snare)):
        want = [sampler.mono(sampler.load_shot(s, pal, cache=tmp_path / "cache")[0])
                for s in pal.kits["jazz"].lanes[lane].layers["med"]]
        assert len(got) == len(want) == 1 and np.array_equal(got[0], want[0]), lane
    assert pal.kits["jazz"].lanes["shaker"].layers["med"][0].note == 82
    choke_only = K.reference_from_palette(DEV, "jazz", shots=False)
    assert choke_only.choke == ref.choke and choke_only.shaker == ()


# --------------------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("edit, message", [
    (lambda s: s.replace("license: CC0 1.0\n", "license: CC0 1.0\ncolour: red\n"), "unknown key"),
    (lambda s: s.replace("  ohat:", "  ohat_x:"), "ohat"),
    (lambda s: s.replace("rr: neighbours", "rr: nearby"), "neighbours"),
    (lambda s: s.replace("vl: {soft: 2, med: 3, hard: 4}", "vl: {soft: 2, med: 3}"), "hard"),
    (lambda s: s.replace("{mic}_vl{vl}.wav", "{mic}_vl{vl}_{take}.wav"), "take"),
    (lambda s: s.replace("id: kit_synth", "id: Kit-Synth"), "id"),
])
def test_a_bad_spec_is_refused_naming_the_problem(tmp_path, edit, message):
    spec = tmp_path / "bad.yaml"
    spec.write_text(edit(SPEC.replace("BORROWED", "false")))
    big = tmp_path / "big"
    with pytest.raises(K.KitError, match=message):
        K.import_kit(K.load_spec(spec), big, reference=synth_reference())


def test_missing_files_other_rates_and_unlocked_changes_are_refused(tmp_path):
    big, spec = synth_kit(tmp_path)
    ref = synth_reference()
    (big / "synth/snare/b_vl5.wav").unlink()
    with pytest.raises(K.KitError, match="synth/snare/b_vl5.wav is missing"):
        K.import_kit(K.load_spec(spec), big, reference=ref)
    big, spec = synth_kit(tmp_path / "2")
    write(big / "synth/rim/c_vl2_rr1.wav", shot_signal(3, lead=50), 32000)
    with pytest.raises(K.KitError, match="32000 Hz"):
        K.import_kit(K.load_spec(spec), big, reference=ref)
    big, spec = synth_kit(tmp_path / "3")
    rows = ["path\tsize\tgit_sha1\tsha256\tcommit"]
    for p in sorted((big / "synth").rglob("*.wav")):
        rows.append(f"{p.relative_to(big / 'synth').as_posix()}\t{p.stat().st_size}\t-\t"
                    f"{hashlib.sha256(p.read_bytes()).hexdigest()}\t-")
    (big / "_locks").mkdir()
    (big / "_locks" / "synth.tsv").write_text("\n".join(rows) + "\n")
    report = K.import_kit(K.load_spec(spec), big, reference=ref).report
    assert report["locks"] == {"synth": f"{len(rows) - 1} files match"} and len(rows) - 1 == 40
    write(big / "synth/hat/c_vl2_rr2.wav", shot_signal(4, lead=70), 44100)
    with pytest.raises(K.KitError, match="hat/c_vl2_rr2.wav does not match its sha256 in .*synth.tsv"):
        K.import_kit(K.load_spec(spec), big, reference=ref)


# ------------------------------------------------------------------------------ the shipped kit specs
KITS = ROOT / "shootouts" / "kits"
TABLE = {   # SHOOTOUTS.md §7.4 kit maps: lane -> (velocity picks, round robins per layer) and the mic premix
    "kit_virtuosity": ({"kick": ((1, 2, 4), 4), "snare": ((9, 22, 32), 3), "rim": ((4, 10, 14), 3),
                        "hat": ((1, 2, 4), 4), "ohat": ((1, 2, 4), 3), "shaker": (None, 14)},
                       {"kick": ["kickmic", "snaremic", "oh"], "shaker": ["close", "oh"]}),
    "kit_swirly": ({"kick": ((6, 13, 20), 4), "snare": ((3, 7, 11), 4), "rim": ((3, 6, 9), 4),
                    "hat": ((3, 7, 11), 4), "ohat": ((4, 8, 13), 4), "shaker": (None, 4)},
                   {"kick": ["beater", "reso"], "snare": ["top", "btm"], "rim": ["top", "btm"], "hat": [""],
                    "shaker": [""]}),
}


@pytest.mark.parametrize("kit", sorted(TABLE))
def test_the_shipped_specs_carry_the_plans_kit_maps(kit):
    spec = K.load_spec(KITS / f"{kit}.yaml")
    lanes, mics = TABLE[kit]
    assert spec.id == kit and set(spec.lanes) == set(LANES)
    for lane, (picks, rr) in lanes.items():
        ls = spec.lanes[lane]
        shots = K.expand(ls)
        assert all(len(shots[layer]) == rr for layer in LAYERS), lane
        if picks is not None:
            assert tuple(ls.vl[layer] for layer in LAYERS) == picks, lane
        assert ls.borrowed == (lane == "shaker"), lane
        if lane in mics:
            assert [m.name for m in ls.mics] == mics[lane] and all(m.db == 0.0 for m in ls.mics), lane
    shaker = [s.member for s in K.expand(spec.lanes["shaker"])["med"]]
    if kit == "kit_virtuosity":
        assert shaker == [f"shaker/{d}_rr{r}.wav" for r in range(1, 8) for d in "DU"]
        assert [m.name for m in spec.lanes["snare"].mics] == ["kickmic", "snaremic", "oh"]
    else:
        assert shaker == ["shaker/D_rr2.wav", "shaker/U_rr1.wav", "shaker/D_rr3.wav", "shaker/U_rr4.wav"]
        files = K.expand(spec.lanes["shaker"])["soft"][0].files
        assert [f for f, _ in files] == ["vcsl/Idiophones/Struck Idiophones/Shaker, Large/LShaker_Shake1D_rr2_Mid.wav"]


# ------------------------------------------------------------------------------------ real banks (cluster)
def big_or_skip() -> Path:
    for var in ("LOFI_ASSETS_BIG", "LOFI_ASSETS"):
        value = os.environ.get(var)
        if value and Path(value).is_dir():
            return Path(value)
    pytest.skip("LOFI_ASSETS_BIG (or LOFI_ASSETS) does not name the assets/big folder")


@pytest.mark.cluster
@pytest.mark.parametrize("kit", sorted(TABLE))
def test_the_real_kits_import_and_load(kit, tmp_path):
    big = big_or_skip()
    spec = K.load_spec(KITS / f"{kit}.yaml")
    # The picks follow CD-33 from the velocity layers on disk.
    for lane, ls in spec.lanes.items():
        if ls.vl:
            assert {layer: ls.vl[layer] for layer in LAYERS} == K.cd33_layers(K.layer_count(ls, big)), lane
    ref = K.reference_from_palette(DEV, "jazz", cache=tmp_path / "cache")
    imp = K.import_kit(spec, big, reference=ref)
    again = K.import_kit(spec, big, reference=ref)
    assert again.data == imp.data
    out = tmp_path / f"{kit}.tar"
    out.write_bytes(imp.data)
    blocks = imp.blocks(str(out))
    assert blocks["choke"] == DEV_CHOKE
    with tarfile.open(out) as t:
        for m in t.getmembers():
            x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32", always_2d=True)
            assert sr == 44100 and x.shape[1] == 1 and m.mtime == 0 and m.mode == 0o644
            thr = np.abs(x).max() * 10 ** ((-20 if m.name.startswith("shaker/") else -30) / 20)
            assert abs(int(np.flatnonzero(np.abs(x[:, 0]) >= thr)[0]) - 44) <= 1, m.name
    doc = copy.deepcopy(DEV_DOC)
    doc["sources"].update(blocks["sources"])
    doc["kits"]["jazz"] = {"bass": doc["kits"]["jazz"]["bass"], "choke": blocks["choke"], "lanes": blocks["lanes"]}
    (tmp_path / "pal.yaml").write_text(yaml.safe_dump(doc))
    pal = load_palette(tmp_path / "pal.yaml", verify=True)
    assert all(pal.kits["jazz"].lanes[lane].layers[layer] for lane in LANES for layer in LAYERS)
    shots = kit_shots(pal, "jazz")
    # The borrowed shaker reproduces the GU shaker-to-snare difference on the grid.
    want = K.grid_lufs(ref.shaker, ref.gain) - K.grid_lufs(ref.snare, ref.gain)
    got = (K.grid_lufs(shots.shots[("shaker", "med")], ref.gain, db=imp.offsets["shaker"])
           - K.grid_lufs(shots.shots[("snare", "med")], ref.gain))
    assert abs(got - want) < 0.1
    r = imp.report
    print(f"\n{kit}: sha256 {imp.sha256}, {len(imp.data):,} bytes, {r['members']} one-shots; lead-in trimmed "
          f"{r['lead_in']['min']}/{r['lead_in']['median']}/{r['lead_in']['max']} samples (min/median/max); "
          f"shaker offset {imp.offsets['shaker']} dB; GU difference {want:.3f} LU, kit {got:.3f} LU")
    for lane in LANES:
        print(f"  {lane}: " + ", ".join(f"{layer} {len(imp.lanes[lane][layer])}" for layer in LAYERS))


@pytest.mark.cluster
def test_the_kit_files_hold_the_rates_and_channels_the_plan_read_from_them():
    # SHOOTOUTS.md §7.4: Virtuosity's kickmic and snaremic files are mono, its oh files stereo, all 48 kHz, its
    # perc shakers 44.1 kHz; Swirly is mono 44.1 kHz; the VCSL shakers are stereo 44.1 kHz.
    big = big_or_skip()
    seen: dict[str, set] = {}
    for kit in TABLE:
        for ls in K.load_spec(KITS / f"{kit}.yaml").lanes.values():
            for shots in K.expand(ls).values():
                for shot in shots:
                    for rel, _ in shot.files:
                        info = sf.info(str(big / rel))
                        parts = rel.split("/")
                        where = "/".join(parts[2:4]) if parts[2] == "perc" else parts[0] if kit == "kit_swirly" \
                            else parts[2]
                        seen.setdefault(where, set()).add((info.samplerate, info.channels))
    assert seen == {"kickmic": {(48000, 1)}, "snaremic": {(48000, 1)}, "oh": {(48000, 2)},
                    "perc/close": {(44100, 1)}, "perc/oh": {(44100, 2)},
                    "karoryfer-swirly-drums": {(44100, 1)}, "vcsl": {(44100, 2)}}
    assert to_44k(np.zeros(48000), 48000).shape == (44100,)
