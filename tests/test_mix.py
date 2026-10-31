"""The bus mix (instrument.md §4): the palette's pedalboard chains, the sax_warm swap, the faders and the
summing order, block-size invariance (why one fixed BUFFER is safe across cycle boundaries), and the
bus-gain calibration helper."""
from types import MappingProxyType

import numpy as np
import pytest

from helpers_compile import dev_palette
from helpers_render import noise
from lofi_rl.palette import Mix
from lofi_rl.render.mix import BUFFER, BUSES, MixError, board, bus_chain, calibrated_gains, mix_buses, process

SR = 44100


def _signal(seconds: float = 1.5) -> np.ndarray:
    """Decaying noise bursts every 0.25 s: enough level and gaps to move compressors and reverbs."""
    x = noise(2, int(seconds * SR), 7).astype(np.float64)
    t = np.arange(x.shape[1]) % (SR // 4)
    return (x * np.exp(-t / 2000.0) * 1.5).astype(np.float32)


def test_every_chain_gives_the_same_bits_at_any_block_size():
    mix = dev_palette().mix
    x = _signal()
    chains = {b: mix.buses[b] for b in mix.buses} | {"master": mix.master}
    for name, chain in chains.items():
        want = board(chain)(x, SR, buffer_size=BUFFER, reset=True)
        for size in (64, 1000, 4096 + 64):
            got = board(chain)(x, SR, buffer_size=size, reset=True)
            assert np.array_equal(got, want), (name, size)
        assert np.array_equal(process(chain, x), want)


def test_sax_warm_replaces_the_sax_reverb():
    mix = dev_palette().mix
    names = [p.name for p in bus_chain(mix, "sax")]
    assert names == ["HighpassFilter", "PeakFilter", "Compressor", "PeakFilter", "Reverb"]
    warm = Mix(mix.bus_gain_db, mix.bus_target_lu, True, mix.buses, mix.master)
    chain = bus_chain(warm, "sax")
    assert [p.name for p in chain] == names[:4] + ["HighShelfFilter", "LowpassFilter", "Reverb"]
    assert chain[-1] == mix.buses["sax_warm"][-1] and dict(chain[-1].params)["room_size"] == 0.35
    assert bus_chain(warm, "keys") == mix.buses["keys"]


def _flat_mix(gains: dict[str, float]) -> Mix:
    empty = MappingProxyType({b: () for b in ("sax", "sax_warm", "keys", "bass", "drums")})
    return Mix(MappingProxyType(gains), MappingProxyType({b: 0.0 for b in BUSES}), False, empty, ())


def test_faders_scale_each_bus_and_the_buses_sum_in_float64():
    stems = {b: noise(2, 2000, i) for i, b in enumerate(BUSES)}
    gains = {"sax": 0.0, "keys": -6.0, "bass": 3.0, "drums": -1.5}
    out = mix_buses(stems, _flat_mix(gains))
    want = sum(stems[b].astype(np.float64) * 10 ** (gains[b] / 20) for b in BUSES).astype(np.float32)
    assert np.array_equal(out.music, want)
    assert all(np.array_equal(out.buses[b], stems[b]) for b in BUSES)


def test_the_mix_does_not_depend_on_the_thread_count():
    stems = {b: _signal(0.5) * np.float32(0.3 + 0.1 * i) for i, b in enumerate(BUSES)}
    mix = dev_palette().mix
    a, b = mix_buses(stems, mix, workers=1), mix_buses(stems, mix, workers=4)
    assert np.array_equal(a.music, b.music)


def test_the_mix_refuses_missing_or_mismatched_stems():
    stems = {b: noise(2, 100, 1) for b in BUSES}
    with pytest.raises(MixError, match="missing"):
        mix_buses({b: stems[b] for b in BUSES[:3]}, dev_palette().mix)
    with pytest.raises(MixError, match="same length"):
        mix_buses(dict(stems, drums=noise(2, 99, 1)), dev_palette().mix)


def test_calibrated_gains_put_median_bus_loudness_on_target():
    targets = dict(dev_palette().mix.bus_target_lu)
    rows = [{"sax": -35.0, "keys": -33.0, "bass": -40.0, "drums": -49.0},
            {"sax": -34.0, "keys": -34.0, "bass": -39.0, "drums": -47.0},
            {"sax": -36.0, "keys": None, "bass": -41.0, "drums": -48.0}]
    gains = calibrated_gains(rows, targets)
    assert gains == {"sax": 0.0, "keys": -7.5, "bass": 2.0, "drums": 11.5}
    after = [{b: r[b] + gains[b] for b in BUSES if r[b] is not None} for r in rows]
    medians = {b: float(np.median([r[b] for r in after if b in r])) for b in BUSES}
    for b in BUSES:
        assert medians[b] - medians["sax"] == pytest.approx(targets[b] - targets["sax"], abs=0.01)
