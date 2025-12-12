import numpy as np
import pytest

from channel.frequency_offset import FrequencyOffset


def _complex_randn(rng, shape, dtype=np.complex64):
    re = rng.standard_normal(shape).astype(np.float32)
    im = rng.standard_normal(shape).astype(np.float32)
    return (re + 1j * im).astype(dtype)


# =======================================================
# NEGATIVNI SCENARIJI (očekujemo grešku)
# =======================================================

def test_apply_rejects_non_complex_float():
    fo = FrequencyOffset(freq_offset_hz=100.0, sample_rate_hz=1000.0)
    x = np.ones(16, dtype=np.float32)
    with pytest.raises(ValueError):
        fo.apply(x)


def test_apply_rejects_non_complex_int():
    fo = FrequencyOffset(freq_offset_hz=100.0, sample_rate_hz=1000.0)
    x = np.ones(16, dtype=np.int32)
    with pytest.raises(ValueError):
        fo.apply(x)


def test_apply_rejects_0d_input():
    fo = FrequencyOffset(freq_offset_hz=100.0, sample_rate_hz=1000.0)
    x0 = np.asarray(1 + 1j)  # ndim = 0
    with pytest.raises(ValueError):
        fo.apply(x0)


# =======================================================
# POZITIVNI SCENARIJI
# =======================================================

def test_apply_empty_1d_returns_empty():
    fo = FrequencyOffset(freq_offset_hz=100.0, sample_rate_hz=1000.0)
    x = np.ones((0,), dtype=np.complex64)
    y = fo.apply(x)
    assert y.shape == (0,)
    assert y.dtype == np.complex64


def test_zero_offset_is_identity_1d():
    fo = FrequencyOffset(freq_offset_hz=0.0, sample_rate_hz=10_000.0, initial_phase_rad=0.0)
    x = _complex_randn(np.random.default_rng(0), (4096,), np.complex64)
    y = fo.apply(x)
    assert np.allclose(y, x)


def test_zero_offset_with_nonzero_initial_phase_is_constant_rotation_nd():
    phi0 = 1.2
    fo = FrequencyOffset(freq_offset_hz=0.0, sample_rate_hz=10_000.0, initial_phase_rad=phi0)

    x = _complex_randn(np.random.default_rng(1), (2, 3, 256), np.complex64)
    y = fo.apply(x)

    expected = x * np.exp(1j * phi0).astype(np.complex64)
    assert np.allclose(y, expected, atol=1e-6)



def test_preserves_shape_and_dtype_complex64():
    fo = FrequencyOffset(freq_offset_hz=500.0, sample_rate_hz=1.0e6, initial_phase_rad=0.1)
    x = np.ones((2, 128), dtype=np.complex64)
    y = fo.apply(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_preserves_shape_and_dtype_complex128():
    fo = FrequencyOffset(freq_offset_hz=500.0, sample_rate_hz=1.0e6, initial_phase_rad=0.1)
    x = np.ones((2, 128), dtype=np.complex128)
    y = fo.apply(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_rotation_preserves_magnitude():
    fs = 1000.0
    df = 123.0
    fo = FrequencyOffset(freq_offset_hz=df, sample_rate_hz=fs, initial_phase_rad=0.0)

    x = _complex_randn(np.random.default_rng(2), (20_000,), np.complex64)
    y = fo.apply(x)

    # rotacija: |y| == |x|
    assert np.allclose(np.abs(y), np.abs(x), rtol=1e-6, atol=1e-6)


def test_initial_phase_applied_to_first_sample_for_unit_input():
    fs = 1000.0
    df = 100.0
    phi0 = 0.7
    fo = FrequencyOffset(freq_offset_hz=df, sample_rate_hz=fs, initial_phase_rad=phi0)

    x = np.ones(8, dtype=np.complex64)
    y = fo.apply(x)

    expected0 = np.exp(1j * phi0).astype(np.complex64)
    assert y[0] == pytest.approx(expected0, rel=1e-6, abs=1e-6)


def test_known_rotation_for_small_n():
    """
    Provjera eksplicitne formule za mali N:
      r[n] = exp(j*(2*pi*df*n/fs + phi0)) za x[n]=1
    """
    fs = 100.0
    df = 5.0
    phi0 = 0.3
    fo = FrequencyOffset(freq_offset_hz=df, sample_rate_hz=fs, initial_phase_rad=phi0)

    N = 10
    x = np.ones(N, dtype=np.complex64)
    y = fo.apply(x)

    n = np.arange(N, dtype=np.float64)
    expected = np.exp(1j * (2.0 * np.pi * df * (n / fs) + phi0)).astype(np.complex64)

    assert np.allclose(y, expected, atol=1e-6)


def test_phase_continuity_across_multiple_apply_calls():
    fs = 1000.0
    df = 50.0

    fo = FrequencyOffset(freq_offset_hz=df, sample_rate_hz=fs, initial_phase_rad=0.0)
    y1 = fo.apply(np.ones(10, dtype=np.complex64))
    y2 = fo.apply(np.ones(15, dtype=np.complex64))

    fo2 = FrequencyOffset(freq_offset_hz=df, sample_rate_hz=fs, initial_phase_rad=0.0)
    y_all = fo2.apply(np.ones(25, dtype=np.complex64))

    assert np.allclose(y1, y_all[:10], atol=1e-6)
    assert np.allclose(y2, y_all[10:], atol=1e-6)


def test_reset_restarts_phase_progression():
    fs = 1000.0
    df = 50.0

    fo = FrequencyOffset(freq_offset_hz=df, sample_rate_hz=fs, initial_phase_rad=0.0)
    x = np.ones(32, dtype=np.complex64)

    y1 = fo.apply(x)
    fo.reset()
    y2 = fo.apply(x)

    assert np.allclose(y1, y2, atol=1e-6)


def test_broadcasting_last_dimension_is_time_axis():
    """
    Provjerava da se rotacija primjenjuje samo duž zadnje dimenzije,
    a sve ostale dimenzije se broadcastuju.
    """
    fs = 1000.0
    df = 25.0
    fo = FrequencyOffset(freq_offset_hz=df, sample_rate_hz=fs, initial_phase_rad=0.0)

    x = np.ones((4, 3, 16), dtype=np.complex64)
    y = fo.apply(x)

    # sve "grupe" po prvim dimenzijama trebaju biti identične, jer je ulaz svuda 1
    assert np.allclose(y[0, 0, :], y[3, 2, :], atol=1e-6)


def test_rotation_vector_zero_or_negative_num_samples_returns_empty():
    """
    Direktan test interne helper metode (dozvoljeno u unit testu).
    """
    fo = FrequencyOffset(freq_offset_hz=10.0, sample_rate_hz=1000.0)
    r0 = fo._rotation_vector(0)
    rneg = fo._rotation_vector(-5)
    assert r0.shape == (0,)
    assert rneg.shape == (0,)
    assert r0.dtype == np.complex64
    assert rneg.dtype == np.complex64


def test_rotation_vector_advances_internal_index():
    fo = FrequencyOffset(freq_offset_hz=10.0, sample_rate_hz=1000.0)
    assert fo._sample_index == 0
    _ = fo._rotation_vector(7)
    assert fo._sample_index == 7
    _ = fo._rotation_vector(3)
    assert fo._sample_index == 10
