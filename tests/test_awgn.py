import numpy as np
import pytest

from channel.awgn_channel import AWGNChannel


# -----------------------------
# Helperi
# -----------------------------
def _measured_snr_db(x: np.ndarray, y: np.ndarray) -> float:
    """Procjena SNR-a iz izlaza: SNR = P_signal / P_noise."""
    noise = y - x
    ps = np.mean(np.abs(x) ** 2)
    pn = np.mean(np.abs(noise) ** 2)
    return 10.0 * np.log10(ps / pn)


# -----------------------------
# UNHAPPY PATHS (greške)
# -----------------------------
def test_init_rejects_nan():
    with pytest.raises(ValueError):
        AWGNChannel(snr_db=np.nan)


def test_init_rejects_inf():
    with pytest.raises(ValueError):
        AWGNChannel(snr_db=np.inf)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_apply_rejects_non_complex_input(dtype):
    ch = AWGNChannel(snr_db=10.0, seed=0)
    x = np.ones(1000, dtype=dtype)
    with pytest.raises(ValueError):
        _ = ch.apply(x)


def test_apply_rejects_zero_power_signal():
    ch = AWGNChannel(snr_db=10.0, seed=0)
    x = np.zeros(2048, dtype=np.complex64)
    with pytest.raises(ValueError):
        _ = ch.apply(x)


def test_apply_rejects_nan_power_signal():
    ch = AWGNChannel(snr_db=10.0, seed=0)
    x = np.ones(1024, dtype=np.complex64)
    x[0] = np.nan + 1j * 0.0
    with pytest.raises(ValueError):
        _ = ch.apply(x)


def test_apply_rejects_inf_power_signal():
    ch = AWGNChannel(snr_db=10.0, seed=0)
    x = np.ones(1024, dtype=np.complex64)
    x[0] = np.inf + 1j * 0.0
    with pytest.raises(ValueError):
        _ = ch.apply(x)


# -----------------------------
# HAPPY PATHS (radi kako treba)
# -----------------------------
def test_output_shape_is_preserved_1d():
    ch = AWGNChannel(snr_db=15.0, seed=1)
    x = (np.random.randn(5000) + 1j * np.random.randn(5000)).astype(np.complex64)
    y = ch.apply(x)
    assert y.shape == x.shape


def test_output_shape_is_preserved_2d():
    ch = AWGNChannel(snr_db=15.0, seed=1)
    x = (np.random.randn(2, 4096) + 1j * np.random.randn(2, 4096)).astype(np.complex64)
    y = ch.apply(x)
    assert y.shape == x.shape


def test_output_shape_is_preserved_3d():
    ch = AWGNChannel(snr_db=15.0, seed=1)
    x = (np.random.randn(2, 3, 2048) + 1j * np.random.randn(2, 3, 2048)).astype(np.complex64)
    y = ch.apply(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_output_dtype_is_preserved(dtype):
    ch = AWGNChannel(snr_db=20.0, seed=2)
    x = (np.random.randn(3000) + 1j * np.random.randn(3000)).astype(dtype)
    y = ch.apply(x)
    assert y.dtype == x.dtype


def test_apply_does_not_modify_input_in_place():
    ch = AWGNChannel(snr_db=10.0, seed=3)
    x = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
    x_copy = x.copy()
    _ = ch.apply(x)
    assert np.array_equal(x, x_copy), "Ulazni signal se ne smije mijenjati in-place."


def test_same_seed_same_result_for_same_input():
    x = (np.random.randn(8192) + 1j * np.random.randn(8192)).astype(np.complex64)

    ch1 = AWGNChannel(snr_db=12.0, seed=123)
    ch2 = AWGNChannel(snr_db=12.0, seed=123)

    y1 = ch1.apply(x)
    y2 = ch2.apply(x)

    assert np.array_equal(y1, y2), "Sa istim seed-om rezultat treba biti identičan."


def test_different_seed_gives_different_result():
    x = (np.random.randn(8192) + 1j * np.random.randn(8192)).astype(np.complex64)

    ch1 = AWGNChannel(snr_db=12.0, seed=1)
    ch2 = AWGNChannel(snr_db=12.0, seed=2)

    y1 = ch1.apply(x)
    y2 = ch2.apply(x)

    assert not np.array_equal(y1, y2), "Različit seed treba dati različit šum."


def test_measured_snr_is_close_to_target():
    # Veći N da statistika “sjedne”
    x = (np.random.randn(200_000) + 1j * np.random.randn(200_000)).astype(np.complex64)
    target_snr_db = 10.0
    ch = AWGNChannel(snr_db=target_snr_db, seed=0)
    y = ch.apply(x)

    snr_est = _measured_snr_db(x, y)
    assert abs(snr_est - target_snr_db) < 0.4, f"SNR procjena {snr_est:.2f} dB odstupa previše."


def test_noise_has_approximately_zero_mean():
    x = (np.random.randn(200_000) + 1j * np.random.randn(200_000)).astype(np.complex64)
    ch = AWGNChannel(snr_db=5.0, seed=0)
    y = ch.apply(x)
    n = y - x

    assert abs(np.mean(n.real)) < 5e-3
    assert abs(np.mean(n.imag)) < 5e-3


def test_noise_real_imag_have_similar_variance():
    x = (np.random.randn(200_000) + 1j * np.random.randn(200_000)).astype(np.complex64)
    ch = AWGNChannel(snr_db=7.0, seed=0)
    y = ch.apply(x)
    n = y - x

    vr = np.var(n.real)
    vi = np.var(n.imag)
    ratio = vr / vi if vi > 0 else np.inf
    assert 0.9 < ratio < 1.1, f"Varijanse real/imag nisu slične: vr={vr}, vi={vi}"


def test_negative_snr_is_supported_and_adds_lots_of_noise():
    x = (np.random.randn(200_000) + 1j * np.random.randn(200_000)).astype(np.complex64)
    ch = AWGNChannel(snr_db=-5.0, seed=0)  # šum jači od signala
    y = ch.apply(x)

    snr_est = _measured_snr_db(x, y)
    assert abs(snr_est - (-5.0)) < 0.6


def test_high_snr_gives_small_noise_power():
    x = (np.random.randn(200_000) + 1j * np.random.randn(200_000)).astype(np.complex64)
    ch = AWGNChannel(snr_db=40.0, seed=0)
    y = ch.apply(x)

    n = y - x
    pn = np.mean(np.abs(n) ** 2)
    ps = np.mean(np.abs(x) ** 2)
    assert pn < ps * 1e-3, "Za 40 dB očekujemo jako malu snagu šuma."
