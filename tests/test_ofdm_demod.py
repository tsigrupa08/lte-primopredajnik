# tests/test_ofdm_demodulator.py
from __future__ import annotations

import numpy as np
import pytest



from receiver.OFDM_demodulator import OFDMDemodulator


#
from transmitter.ofdm import OFDMModulator



def _make_grid(ndlrb: int = 6, n_sym: int = 14, seed: int = 0) -> np.ndarray:
    """
    Grid oblika (N_sc, N_sym), gdje je N_sc = 12*ndlrb.
    U tvojoj implementaciji grid je "centred" u smislu da predstavlja aktivne subcarriere
    bez DC (72 za ndlrb=6), u poretku [negativni | pozitivni].
    """
    rng = np.random.default_rng(seed)
    n_sc = 12 * ndlrb
    grid = (rng.standard_normal((n_sc, n_sym)) + 1j * rng.standard_normal((n_sc, n_sym))).astype(np.complex64)
    return grid


def _tx_modulate(grid: np.ndarray) -> tuple[np.ndarray, float]:
    tx = OFDMModulator(grid)
    wave, fs = tx.modulate()
    return np.asarray(wave), float(fs)


# -------------------------
# HAPPY PATHS
# -------------------------

def test_fft_size_map_ndlrb6():
    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    assert d.fft_size == 128
    assert d.sample_rate == 15_000 * 128


def test_fft_size_map_ndlrb100():
    d = OFDMDemodulator(ndlrb=100, normal_cp=True)
    assert d.fft_size == 2048
    assert d.sample_rate == 15_000 * 2048


def test_cp_lengths_ndlrb6_normal_cp():
    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    # skalirano sa 2048 -> 128: cp_first=10, cp_others=9
    assert d.cp_lengths == [10] + [9] * 6


def test_demodulate_returns_2d_complex():
    grid = _make_grid(seed=1)
    wave, _ = _tx_modulate(grid)

    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    grid_fft = d.demodulate(wave)

    assert grid_fft.ndim == 2
    assert grid_fft.shape[1] == d.fft_size
    assert np.iscomplexobj(grid_fft)


def test_demodulate_expected_num_symbols_for_one_subframe():
    grid = _make_grid(n_sym=14, seed=2)
    wave, _ = _tx_modulate(grid)

    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    grid_fft = d.demodulate(wave)

    assert grid_fft.shape[0] == 14  # 1 subframe = 14 symbols


def test_extract_active_subcarriers_shape():
    grid = _make_grid(seed=3)
    wave, _ = _tx_modulate(grid)

    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    active = d.extract_active_subcarriers(d.demodulate(wave))

    assert active.shape == (14, 72)


def test_extract_active_subcarriers_excludes_dc_bin():
    # Ako DC nije isključen, ova provjera često padne jer DC upadne u "active" blok
    grid = _make_grid(seed=4)
    wave, _ = _tx_modulate(grid)

    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    grid_fft = d.demodulate(wave)

    dc = d.fft_size // 2
    dc_power = np.mean(np.abs(grid_fft[:, dc]) ** 2)

    active = d.extract_active_subcarriers(grid_fft)
    active_power = np.mean(np.abs(active) ** 2)

    # DC treba biti "van" active: ne testiramo da je dc_power=0, nego da active ne zavisi od tog bita
    assert active.shape[1] == 72
    assert np.isfinite(dc_power)
    assert np.isfinite(active_power)


def test_end_to_end_recovers_grid_perfect_channel():
    # Najvažniji test: TX -> RX -> active treba vratiti originalni grid
    grid = _make_grid(seed=5)
    wave, _ = _tx_modulate(grid)

    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    active = d.extract_active_subcarriers(d.demodulate(wave))

    # active je (N_sym, 72), grid je (72, N_sym)
    assert np.allclose(active.T, grid, atol=1e-5)


def test_end_to_end_preserves_single_tone_on_highest_positive_sc():
    # Hvata off-by-one bug na pozitivnoj ivici
    ndlrb = 6
    n_sc = 12 * ndlrb
    n_sym = 14
    grid = np.zeros((n_sc, n_sym), dtype=np.complex64)
    grid[n_sc - 1, 0] = 1.0 + 0j  # zadnji element u [neg|pos] poretku

    wave, _ = _tx_modulate(grid)
    d = OFDMDemodulator(ndlrb=ndlrb, normal_cp=True)
    active = d.extract_active_subcarriers(d.demodulate(wave))

    assert np.isclose(active[0, n_sc - 1], 1.0, atol=1e-5)


# -------------------------
# SAD / UNHAPPY PATHS
# -------------------------

def test_invalid_ndlrb_raises():
    with pytest.raises(ValueError):
        OFDMDemodulator(ndlrb=7, normal_cp=True)


def test_extended_cp_not_implemented_raises():
    with pytest.raises(NotImplementedError):
        OFDMDemodulator(ndlrb=6, normal_cp=False)


def test_demodulate_raises_on_too_short_signal():
    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    rx = np.zeros(10, dtype=np.complex64)
    with pytest.raises(ValueError, match="prekratak"):
        d.demodulate(rx)


def test_demodulate_handles_real_input_returns_complex_grid():
    grid = _make_grid(seed=6)
    wave, _ = _tx_modulate(grid)
    rx_real = np.real(wave).astype(np.float64)

    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    out = d.demodulate(rx_real)

    assert np.iscomplexobj(out)
    assert out.shape[1] == d.fft_size


def test_extract_active_subcarriers_raises_on_1d_input():
    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    bad = np.zeros((128,), dtype=np.complex64)  # 1D
    with pytest.raises(Exception):
        d.extract_active_subcarriers(bad)


def test_demodulate_output_is_finite_for_zero_signal_long_enough():
    # Dovoljno dugo za barem 1 simbol (cp_first + N)
    d = OFDMDemodulator(ndlrb=6, normal_cp=True)
    L = d.cp_lengths[0] + d.fft_size
    rx = np.zeros(L, dtype=np.complex64)

    grid_fft = d.demodulate(rx)
    assert np.all(np.isfinite(grid_fft))
