import numpy as np
import pytest

from channel.awgn_channel import AWGNChannel
from channel.frequency_offset import FrequencyOffset


def _complex_randn(rng, shape, dtype=np.complex64, scale=1.0):
    re = (rng.standard_normal(shape).astype(np.float32) * scale)
    im = (rng.standard_normal(shape).astype(np.float32) * scale)
    return (re + 1j * im).astype(dtype)


def _estimate_snr_db(x, y):
    n = y - x
    ps = np.mean(np.abs(x) ** 2)
    pn = np.mean(np.abs(n) ** 2)
    return 10.0 * np.log10(ps / pn)


# =======================================================
# NEGATIVNI SCENARIJI (očekujemo greške)
# =======================================================

def test_awgn_init_rejects_nan_inf():
    with pytest.raises(ValueError):
        AWGNChannel(np.nan)
    with pytest.raises(ValueError):
        AWGNChannel(np.inf)
    with pytest.raises(ValueError):
        AWGNChannel(-np.inf)


def test_awgn_apply_rejects_non_complex():
    ch = AWGNChannel(10.0, seed=0)
    with pytest.raises(ValueError):
        ch.apply(np.ones(16, dtype=np.float32))
    with pytest.raises(ValueError):
        ch.apply(np.ones(16, dtype=np.int32))


def test_awgn_apply_rejects_zero_power_signal():
    ch = AWGNChannel(10.0, seed=0)
    x = np.zeros(128, dtype=np.complex64)
    with pytest.raises(ValueError):
        ch.apply(x)


def test_awgn_apply_rejects_nan_inf_in_signal():
    ch = AWGNChannel(10.0, seed=0)

    x = np.ones(32, dtype=np.complex64)
    x[0] = np.nan + 1j * 0.0
    with pytest.raises(ValueError):
        ch.apply(x)

    x = np.ones(32, dtype=np.complex64)
    x[0] = np.inf + 1j * 0.0
    with pytest.raises(ValueError):
        ch.apply(x)


def test_fo_apply_rejects_non_complex():
    fo = FrequencyOffset(freq_offset_hz=100.0, sample_rate_hz=1000.0)
    with pytest.raises(ValueError):
        fo.apply(np.ones(16, dtype=np.float32))


def test_fo_apply_rejects_0d_input():
    fo = FrequencyOffset(freq_offset_hz=100.0, sample_rate_hz=1000.0)
    x0 = np.asarray(1 + 1j)  # ndim = 0
    with pytest.raises(ValueError):
        fo.apply(x0)


# =======================================================
# POZITIVNI SCENARIJI (ispravan rad)
# =======================================================

def test_awgn_preserves_shape_and_dtype_for_various_dims():
    ch = AWGNChannel(10.0, seed=0)

    x1 = (np.ones(100, dtype=np.complex64) + 1j*np.ones(100, dtype=np.complex64))
    y1 = ch.apply(x1)
    assert y1.shape == x1.shape
    assert y1.dtype == x1.dtype

    x2 = (np.ones((2, 128), dtype=np.complex128) + 1j*np.ones((2, 128), dtype=np.complex128))
    y2 = ch.apply(x2)
    assert y2.shape == x2.shape
    assert y2.dtype == x2.dtype

    x3 = (np.ones((2, 3, 64), dtype=np.complex64) + 1j*np.ones((2, 3, 64), dtype=np.complex64))
    y3 = ch.apply(x3)
    assert y3.shape == x3.shape
    assert y3.dtype == x3.dtype


def test_awgn_does_not_modify_input_in_place():
    ch = AWGNChannel(10.0, seed=0)
    x = _complex_randn(np.random.default_rng(1), (2048,), dtype=np.complex64)
    x_copy = x.copy()
    _ = ch.apply(x)
    assert np.array_equal(x, x_copy)


def test_awgn_same_seed_same_output_same_input():
    rng = np.random.default_rng(123)
    x = _complex_randn(rng, (4096,), dtype=np.complex64)

    y1 = AWGNChannel(15.0, seed=42).apply(x)
    y2 = AWGNChannel(15.0, seed=42).apply(x)

    assert np.array_equal(y1, y2)


def test_awgn_rng_advances_between_calls():
    rng = np.random.default_rng(123)
    x = _complex_randn(rng, (4096,), dtype=np.complex64)

    ch = AWGNChannel(15.0, seed=42)
    y1 = ch.apply(x)
    y2 = ch.apply(x)

    assert not np.array_equal(y1, y2)


def test_awgn_measured_snr_close_to_target():
    # statistički test: na većem N treba da bude blizu cilja
    snr_db = 20.0
    rng = np.random.default_rng(2024)
    x = _complex_randn(rng, (200_000,), dtype=np.complex64)

    ch = AWGNChannel(snr_db, seed=7)
    y = ch.apply(x)

    snr_meas = _estimate_snr_db(x, y)
    assert snr_meas == pytest.approx(snr_db, abs=0.6)


def test_awgn_higher_snr_means_lower_noise_power():
    rng = np.random.default_rng(2025)
    x = _complex_randn(rng, (80_000,), dtype=np.complex64)

    y_low = AWGNChannel(0.0, seed=1).apply(x)
    y_high = AWGNChannel(30.0, seed=1).apply(x)

    pn_low = np.mean(np.abs(y_low - x) ** 2)
    pn_high = np.mean(np.abs(y_high - x) ** 2)
    assert pn_high < pn_low


def test_fo_zero_offset_is_identity():
    fo = FrequencyOffset(freq_offset_hz=0.0, sample_rate_hz=10_000.0)
    x = _complex_randn(np.random.default_rng(0), (2, 3, 1024), dtype=np.complex64)
    y = fo.apply(x)
    assert np.allclose(y, x)


def test_fo_preserves_shape_and_dtype():
    fo = FrequencyOffset(freq_offset_hz=500.0, sample_rate_hz=1.0e6, initial_phase_rad=0.1)
    x = np.ones((2, 128), dtype=np.complex64)
    y = fo.apply(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_fo_rotation_preserves_magnitude():
    fs = 1000.0
    df = 123.0
    fo = FrequencyOffset(df, fs)
    x = _complex_randn(np.random.default_rng(0), (10_000,), np.complex64)
    y = fo.apply(x)
    assert np.allclose(np.abs(y), np.abs(x), rtol=1e-6, atol=1e-6)


def test_fo_initial_phase_applied_to_first_sample():
    fs = 1000.0
    df = 100.0
    phi0 = 0.7
    fo = FrequencyOffset(df, fs, initial_phase_rad=phi0)

    x = np.ones(8, dtype=np.complex64)
    y = fo.apply(x)

    expected0 = np.exp(1j * phi0).astype(np.complex64)
    assert y[0] == pytest.approx(expected0, rel=1e-6, abs=1e-6)


def test_fo_phase_continuity_across_multiple_apply_calls():
    fs = 1000.0
    df = 50.0
    fo = FrequencyOffset(df, fs, initial_phase_rad=0.0)

    y1 = fo.apply(np.ones(10, dtype=np.complex64))
    y2 = fo.apply(np.ones(15, dtype=np.complex64))

    fo2 = FrequencyOffset(df, fs, initial_phase_rad=0.0)
    y_all = fo2.apply(np.ones(25, dtype=np.complex64))

    assert np.allclose(y1, y_all[:10], atol=1e-6)
    assert np.allclose(y2, y_all[10:], atol=1e-6)


def test_fo_reset_restarts_phase_progression():
    fs = 1000.0
    df = 50.0
    fo = FrequencyOffset(df, fs, initial_phase_rad=0.0)

    x = np.ones(10, dtype=np.complex64)
    y1 = fo.apply(x)
    fo.reset()
    y2 = fo.apply(x)

    assert np.allclose(y1, y2, atol=1e-6)


# =======================================================
# “CIJELI KANAL” – SPOJENO (kaskada)
# =======================================================

def test_channel_cascade_preserves_shape_dtype():
    rng = np.random.default_rng(1)
    x = _complex_randn(rng, (2, 3, 4096), dtype=np.complex64)

    fo = FrequencyOffset(freq_offset_hz=250.0, sample_rate_hz=1.0e6, initial_phase_rad=0.2)
    awgn = AWGNChannel(snr_db=15.0, seed=123)

    y = awgn.apply(fo.apply(x))

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_channel_freq_offset_then_awgn_snr_close_to_target():
    # FO je rotacija => ne mijenja snagu, AWGN i dalje treba pogoditi SNR
    snr_db = 18.0
    rng = np.random.default_rng(2)
    x = _complex_randn(rng, (200_000,), dtype=np.complex64)

    fo = FrequencyOffset(freq_offset_hz=777.0, sample_rate_hz=1.0e6, initial_phase_rad=0.3)
    awgn = AWGNChannel(snr_db=snr_db, seed=7)

    x_rot = fo.apply(x)
    y = awgn.apply(x_rot)

    snr_meas = _estimate_snr_db(x_rot, y)
    assert snr_meas == pytest.approx(snr_db, abs=0.7)


def test_channel_awgn_then_freq_offset_noise_power_unchanged():
    # rotacija nakon šuma ne mijenja snagu šuma (magnituda ostaje ista)
    rng = np.random.default_rng(3)
    x = _complex_randn(rng, (60_000,), dtype=np.complex64)

    awgn = AWGNChannel(snr_db=10.0, seed=99)
    fo = FrequencyOffset(freq_offset_hz=500.0, sample_rate_hz=1.0e6, initial_phase_rad=0.0)

    y_awgn = awgn.apply(x)
    pn1 = np.mean(np.abs(y_awgn - x) ** 2)

    y2 = fo.apply(y_awgn)
    x2 = fo.apply(x)  # ista rotacija na signal
    pn2 = np.mean(np.abs(y2 - x2) ** 2)

    assert pn2 == pytest.approx(pn1, rel=1e-3, abs=0.0)
