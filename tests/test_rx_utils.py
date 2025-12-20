# tests/test_rx_utils.py
import numpy as np
import pytest

from receiver.utils import RxUtils, RxValidationConfig


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture()
def utils_default() -> RxUtils:
    return RxUtils()  # default config


@pytest.fixture()
def utils_len_128() -> RxUtils:
    cfg = RxValidationConfig(min_num_samples=128, require_complex=True)
    return RxUtils(cfg)


@pytest.fixture()
def complex_sig_1024() -> np.ndarray:
    rng = np.random.default_rng(123)
    x = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)
    return x


# ---------------------------------------------------------------------
# Tests: ensure_1d_time_axis
# ---------------------------------------------------------------------
def test_ensure_1d_time_axis_accepts_1d(utils_default: RxUtils, complex_sig_1024: np.ndarray):
    y = utils_default.ensure_1d_time_axis(complex_sig_1024)
    assert y.ndim == 1
    assert y.shape == (1024,)


def test_ensure_1d_time_axis_flattens_1_by_n(utils_default: RxUtils, complex_sig_1024: np.ndarray):
    x = complex_sig_1024.reshape(1, -1)
    y = utils_default.ensure_1d_time_axis(x)
    assert y.ndim == 1
    assert y.shape == (1024,)


def test_ensure_1d_time_axis_flattens_n_by_1(utils_default: RxUtils, complex_sig_1024: np.ndarray):
    x = complex_sig_1024.reshape(-1, 1)
    y = utils_default.ensure_1d_time_axis(x)
    assert y.ndim == 1
    assert y.shape == (1024,)

#UNHAPPY PATH
def test_ensure_1d_time_axis_rejects_matrix(utils_default: RxUtils):
    x = np.zeros((2, 3), dtype=np.complex64)
    with pytest.raises(ValueError):
        utils_default.ensure_1d_time_axis(x)

#UNHAPPY PATH
def test_ensure_1d_time_axis_rejects_nd(utils_default: RxUtils):
    x = np.zeros((2, 2, 2), dtype=np.complex64)
    with pytest.raises(ValueError):
        utils_default.ensure_1d_time_axis(x)


# ---------------------------------------------------------------------
# Tests: mean_power + rms
# ---------------------------------------------------------------------
def test_mean_power_and_rms_nonnegative(utils_default: RxUtils, complex_sig_1024: np.ndarray):
    p = utils_default.mean_power(complex_sig_1024)
    r = utils_default.rms(complex_sig_1024)
    assert p > 0.0
    assert r > 0.0
    # RMS^2 približno = power
    assert np.isclose(r * r, p, rtol=1e-6, atol=1e-6)


def test_zero_signal_power(utils_default: RxUtils):
    x = np.zeros(256, dtype=np.complex64)
    assert utils_default.mean_power(x) == 0.0
    assert utils_default.rms(x) == 0.0


# ---------------------------------------------------------------------
# Tests: validate_rx_samples
# ---------------------------------------------------------------------
def test_validate_rx_samples_ok_complex(utils_len_128: RxUtils, complex_sig_1024: np.ndarray):
    y = utils_len_128.validate_rx_samples(complex_sig_1024)
    assert y.shape == complex_sig_1024.shape
    assert np.issubdtype(y.dtype, np.complexfloating)

#UNHAPPY PATH
def test_validate_rx_samples_rejects_too_short(utils_len_128: RxUtils):
    x = (np.ones(64) + 1j * np.ones(64)).astype(np.complex64)
    with pytest.raises(ValueError):
        utils_len_128.validate_rx_samples(x)

#UNHAPPY PATH
def test_validate_rx_samples_rejects_real_if_require_complex(utils_len_128: RxUtils):
    x = np.ones(256, dtype=np.float32)
    with pytest.raises(TypeError):
        utils_len_128.validate_rx_samples(x)


def test_validate_rx_samples_casts_real_to_complex_when_allowed():
    cfg = RxValidationConfig(require_complex=False, cast_real_to_complex=True, min_num_samples=16)
    u = RxUtils(cfg)
    x = np.ones(64, dtype=np.float32)
    y = u.validate_rx_samples(x)
    assert np.issubdtype(y.dtype, np.complexfloating)
    assert np.allclose(y.imag, 0.0)

#UNHAPPY PATH
def test_validate_rx_samples_rejects_nan_inf(utils_default: RxUtils):
    x = (np.ones(256) + 1j * np.ones(256)).astype(np.complex64)
    x[10] = np.nan + 1j * 0.0
    with pytest.raises(ValueError):
        utils_default.validate_rx_samples(x)

#UNHAPPY PATH
def test_validate_rx_samples_rejects_near_zero_power():
    cfg = RxValidationConfig(min_num_samples=16, eps_power=1e-12)
    u = RxUtils(cfg)
    x = (1e-20 * (np.ones(256) + 1j * np.ones(256))).astype(np.complex64)
    with pytest.raises(ValueError):
        u.validate_rx_samples(x)


def test_validate_rx_samples_preserves_shape_nd(utils_default: RxUtils):
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((2, 3, 512)) + 1j * rng.standard_normal((2, 3, 512))).astype(np.complex64)
    y = utils_default.validate_rx_samples(x, min_num_samples=128)
    assert y.shape == x.shape


# ---------------------------------------------------------------------
# Tests: normalize_rms
# ---------------------------------------------------------------------
def test_normalize_rms_targets_1(utils_default: RxUtils, complex_sig_1024: np.ndarray):
    y, scale = utils_default.normalize_rms(complex_sig_1024, target_rms=1.0)
    assert np.isfinite(scale)
    assert scale > 0.0
    assert np.isclose(utils_default.rms(y), 1.0, rtol=1e-5, atol=1e-5)


def test_normalize_rms_custom_target(utils_default: RxUtils, complex_sig_1024: np.ndarray):
    target = 0.25
    y, _ = utils_default.normalize_rms(complex_sig_1024, target_rms=target)
    assert np.isclose(utils_default.rms(y), target, rtol=1e-5, atol=1e-5)

#UNHAPPY PATH
def test_normalize_rms_rejects_nonpositive_target(utils_default: RxUtils, complex_sig_1024: np.ndarray):
    with pytest.raises(ValueError):
        utils_default.normalize_rms(complex_sig_1024, target_rms=0.0)

#UNHAPPY PATH
def test_normalize_rms_rejects_zero_signal(utils_default: RxUtils):
    x = np.zeros(256, dtype=np.complex64)
    with pytest.raises(ValueError):
        utils_default.normalize_rms(x, target_rms=1.0)


# ---------------------------------------------------------------------
# Tests: dB helpers
# ---------------------------------------------------------------------
def test_db2lin_lin2db_roundtrip(utils_default: RxUtils):
    db = 13.0
    lin = utils_default.db2lin(db)
    db2 = utils_default.lin2db(lin)
    assert np.isclose(db2, db, rtol=1e-12, atol=1e-12)

#UNHAPPY PATH
def test_lin2db_rejects_nonpositive_without_floor(utils_default: RxUtils):
    with pytest.raises(ValueError):
        utils_default.lin2db(0.0)


def test_lin2db_uses_floor(utils_default: RxUtils):
    db = utils_default.lin2db(0.0, floor_db=-300.0)
    assert db == -300.0
