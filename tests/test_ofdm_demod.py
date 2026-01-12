# tests/test_ofdm_demodulator.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ------------------------------------------------------------
# PATH FIX
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from receiver.OFDM_demodulator import OFDMDemodulator


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def build_waveform_for_n_symbols(demod: OFDMDemodulator, n_symbols: int, *, seed: int = 0) -> np.ndarray:
    """
    Napravi sintetički OFDM talasni oblik koji će demodulator moći parsirati:
    za svaki simbol ubaci CP + NFFT uzoraka (random kompleksno).
    CP pattern se ponavlja po slotu.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for sym in range(n_symbols):
        cp = demod.cp_lengths[sym % demod.n_symbols_per_slot]
        total = cp + demod.fft_size
        # random kompleksan signal
        x = rng.normal(size=total) + 1j * rng.normal(size=total)
        chunks.append(x.astype(np.complex128))
    return np.concatenate(chunks)


def build_grid_full_pattern(demod: OFDMDemodulator, Ns: int) -> np.ndarray:
    """
    Napravi fftshift-ovan grid_full (NFFT, Ns) tako da:
      - neg dio (ispod DC) ima vrijednost -1j
      - DC je 99+99j
      - pos dio (iznad DC) ima vrijednost +1j
    Ovo služi da se lako provjeri extract_active_subcarriers.
    """
    N = demod.fft_size
    center = N // 2
    g = np.zeros((N, Ns), dtype=np.complex128)

    # ispuni sve:
    g[:, :] = 0.0 + 0.0j
    g[center, :] = 99.0 + 99.0j  # DC

    # negativni binovi u aktivnom opsegu
    half = demod.n_active // 2
    g[center - half : center, :] = -1.0j

    # pozitivni binovi u aktivnom opsegu (preskoci DC)
    g[center + 1 : center + 1 + half, :] = +1.0j
    return g


# ============================================================
# 1) Parametri / inicijalizacija
# ============================================================

@pytest.mark.parametrize(
    "ndlrb, expected_nfft",
    [(6, 128), (15, 256), (25, 512), (50, 1024), (75, 1536), (100, 2048)],
)
def test_fft_size_mapping(ndlrb: int, expected_nfft: int):
    demod = OFDMDemodulator(ndlrb=ndlrb, normal_cp=True)
    assert demod.fft_size == expected_nfft
    assert demod.n_active == 12 * ndlrb
    assert demod.sample_rate == 15_000 * expected_nfft


def test_invalid_ndlrb_raises():
    with pytest.raises(ValueError):
        _ = OFDMDemodulator(ndlrb=7, normal_cp=True)


def test_cp_lengths_normal_cp_has_7_entries():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    assert demod.n_symbols_per_slot == 7
    assert isinstance(demod.cp_lengths, list)
    assert len(demod.cp_lengths) == 7
    assert demod.cp_lengths[0] > 0
    assert all(cp > 0 for cp in demod.cp_lengths)


def test_cp_lengths_extended_cp_has_6_entries():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=False)
    assert demod.n_symbols_per_slot == 6
    assert isinstance(demod.cp_lengths, list)
    assert len(demod.cp_lengths) == 6
    assert len(set(demod.cp_lengths)) == 1  # svi isti u extended CP


# ============================================================
# 2) demodulate() unhappy paths
# ============================================================

def test_demodulate_rejects_non_1d():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    x = np.zeros((10, 2), dtype=np.complex128)
    with pytest.raises(ValueError):
        demod.demodulate(x)


def test_demodulate_rejects_real_input():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    x = np.zeros(1000, dtype=np.float64)
    with pytest.raises(ValueError):
        demod.demodulate(x)


def test_demodulate_raises_if_too_short_for_one_symbol():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    # manje od cp+NFFT za prvi simbol
    cp0 = demod.cp_lengths[0]
    x = np.zeros(cp0 + demod.fft_size - 1, dtype=np.complex128)
    with pytest.raises(ValueError):
        demod.demodulate(x)


# ============================================================
# 3) demodulate() happy paths (shape / broj simbola)
# ============================================================

def test_demodulate_exactly_one_symbol_returns_grid_full_shape():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    w = build_waveform_for_n_symbols(demod, 1, seed=1)
    g = demod.demodulate(w)
    assert g.shape == (demod.fft_size, 1)
    assert np.iscomplexobj(g)


def test_demodulate_multiple_symbols_returns_correct_num_symbols():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    w = build_waveform_for_n_symbols(demod, 5, seed=2)
    g = demod.demodulate(w)
    assert g.shape == (demod.fft_size, 5)


def test_demodulate_ignores_incomplete_tail_samples():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    w = build_waveform_for_n_symbols(demod, 3, seed=3)

    # dodaj "rep" koji nije dovoljan za još jedan simbol
    w2 = np.concatenate([w, np.zeros(demod.fft_size // 2, dtype=np.complex128)])

    g = demod.demodulate(w2)
    assert g.shape == (demod.fft_size, 3)


# ============================================================
# 4) extract_active_subcarriers() unhappy paths
# ============================================================

def test_extract_active_rejects_non_2d():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    g = np.zeros((demod.fft_size,), dtype=np.complex128)
    with pytest.raises(ValueError):
        demod.extract_active_subcarriers(g)


def test_extract_active_rejects_wrong_nfft_rows():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    g = np.zeros((demod.fft_size + 1, 2), dtype=np.complex128)
    with pytest.raises(ValueError):
        demod.extract_active_subcarriers(g)


# ============================================================
# 5) extract_active_subcarriers() happy + DC skip provjera
# ============================================================

def test_extract_active_returns_shape_12ndlrb_by_Ns():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    Ns = 4
    g_full = np.zeros((demod.fft_size, Ns), dtype=np.complex128)
    g_act = demod.extract_active_subcarriers(g_full)
    assert g_act.shape == (12 * demod.ndlrb, Ns)
    assert np.iscomplexobj(g_act)


def test_extract_active_skips_dc_and_keeps_order_neg_then_pos():
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    Ns = 3
    g_full = build_grid_full_pattern(demod, Ns)
    g_act = demod.extract_active_subcarriers(g_full)

    half = demod.n_active // 2
    neg = g_act[:half, :]
    pos = g_act[half:, :]

    # neg dio treba biti -1j, pos dio +1j
    assert np.allclose(neg, -1.0j)
    assert np.allclose(pos, +1.0j)

    # DC (99+99j) ne smije se pojaviti u grid_active
    assert not np.any(np.isclose(g_act.real, 99.0) & np.isclose(g_act.imag, 99.0))


def test_extract_active_consistent_for_different_ndlrb():
    demod = OFDMDemodulator(ndlrb=15, normal_cp=True)  # NFFT=256, n_active=180
    Ns = 2
    g_full = build_grid_full_pattern(demod, Ns)
    g_act = demod.extract_active_subcarriers(g_full)

    assert g_act.shape == (12 * 15, Ns)
    half = (12 * 15) // 2
    assert np.allclose(g_act[:half, :], -1.0j)
    assert np.allclose(g_act[half:, :], +1.0j)
