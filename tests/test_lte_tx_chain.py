# tests/test_lte_tx_chain.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ------------------------------------------------------------
# PATH FIX (da importi rade kad pytest krene iz root-a)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transmitter.LTETxChain import LTETxChain


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def rand_mib(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=24, dtype=np.uint8)


def assert_complex_1d(x: np.ndarray) -> None:
    x = np.asarray(x)
    assert x.ndim == 1
    assert np.iscomplexobj(x)
    assert x.size > 0
    assert np.all(np.isfinite(np.real(x)))
    assert np.all(np.isfinite(np.imag(x)))


# ============================================================
# 1) Konstruktor / property / reset grid
# ============================================================

def test_symbols_per_subframe_normal_cp_is_14():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    assert tx.symbols_per_subframe == 14


def test_symbols_per_subframe_extended_cp_is_12():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=False)
    assert tx.symbols_per_subframe == 12


def test_grid_shape_matches_config_normal_cp():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=2, normal_cp=True)
    # grid: (12*ndlrb, num_subframes*symbols_per_subframe)
    assert tx.grid.shape == (72, 2 * 14)


def test_grid_shape_matches_config_extended_cp():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=3, normal_cp=False)
    assert tx.grid.shape == (72, 3 * 12)


def test_pss_symbol_index_normal_cp():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    assert tx._pss_symbol_index() == 6


def test_pss_symbol_index_extended_cp():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=False)
    assert tx._pss_symbol_index() == 5


def test_pbch_symbol_indices_normal_cp_subframe0():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    assert tx._pbch_symbol_indices_for_subframe(0) == [7, 8, 9, 10]


def test_pbch_symbol_indices_extended_cp_subframe0():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=False)
    assert tx._pbch_symbol_indices_for_subframe(0) == [6, 7, 8, 9]


def test_pbch_symbol_indices_normal_cp_subframe3_offset():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    # base = sf*14 => 42; +7..10 => 49..52
    assert tx._pbch_symbol_indices_for_subframe(3) == [49, 50, 51, 52]


def test_reset_grid_creates_new_grid():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    old = tx.grid
    tx._reset_grid()
    assert tx.grid is not old
    assert tx.grid.shape == old.shape


# ============================================================
# 2) Unhappy paths (validacije)
# ============================================================

@pytest.mark.parametrize("bad_nid2", [-1, 3, 99])
def test_invalid_n_id_2_raises(bad_nid2: int):
    tx = LTETxChain(n_id_2=bad_nid2, ndlrb=6, num_subframes=1, normal_cp=True)
    with pytest.raises(ValueError):
        tx.generate_waveform(mib_bits=None)


def test_reserved_mask_wrong_shape_raises():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    bad_mask = np.zeros((1, 1), dtype=bool)
    with pytest.raises(ValueError):
        tx.generate_waveform(mib_bits=None, reserved_re_mask=bad_mask)


def test_mib_wrong_length_raises():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = np.zeros(23, dtype=np.uint8)
    with pytest.raises(ValueError):
        tx.generate_waveform(mib_bits=mib)


def test_mib_length_24_but_num_subframes_less_than_4_raises():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=3, normal_cp=True)
    mib = rand_mib(1)
    with pytest.raises(ValueError):
        tx.generate_waveform(mib_bits=mib)


def test_reserved_mask_dtype_not_bool_is_still_accepted_if_shape_ok():
    # nije strictno traženo da bude bool (numpy će truthy/falsy), ali shape mora pasati
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mask = np.zeros(tx.grid.shape, dtype=np.uint8)  # shape ok
    mib = rand_mib(2)
    w, fs = tx.generate_waveform(mib_bits=mib, reserved_re_mask=mask)
    assert_complex_1d(w)
    assert fs > 0.0


# ============================================================
# 3) Happy paths (PSS-only)
# ============================================================

def test_generate_waveform_pss_only_runs_and_returns_complex_waveform():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    w, fs = tx.generate_waveform(mib_bits=None)
    assert_complex_1d(w)
    assert fs > 0.0


@pytest.mark.parametrize("nid2", [0, 1, 2])
def test_generate_waveform_pss_only_for_each_nid2(nid2: int):
    tx = LTETxChain(n_id_2=nid2, ndlrb=6, num_subframes=1, normal_cp=True)
    w, fs = tx.generate_waveform()
    assert_complex_1d(w)
    assert fs > 0.0


def test_generate_waveform_resets_grid_each_call_changes_object_identity():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    _ = tx.generate_waveform()
    g1 = tx.grid
    _ = tx.generate_waveform()
    g2 = tx.grid
    assert g1 is not g2


def test_pss_only_grid_not_all_zero_after_generate():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    _w, _fs = tx.generate_waveform()
    # PSS mora unijeti energiju u grid
    assert np.count_nonzero(tx.grid) > 0


# ============================================================
# 4) Happy paths (PSS+PBCH)
# ============================================================

def test_generate_waveform_with_pbch_runs_and_returns_complex_waveform():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = rand_mib(3)
    w, fs = tx.generate_waveform(mib_bits=mib)
    assert_complex_1d(w)
    assert fs > 0.0


@pytest.mark.parametrize("nid2", [0, 1, 2])
def test_generate_waveform_with_pbch_for_each_nid2(nid2: int):
    tx = LTETxChain(n_id_2=nid2, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = rand_mib(10 + nid2)
    w, fs = tx.generate_waveform(mib_bits=mib)
    assert_complex_1d(w)
    assert fs > 0.0


def test_pbch_increases_grid_energy_vs_pss_only():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)

    _ = tx.generate_waveform(mib_bits=None)
    e_pss = float(np.sum(np.abs(tx.grid) ** 2))

    mib = rand_mib(5)
    _ = tx.generate_waveform(mib_bits=mib)
    e_pbch = float(np.sum(np.abs(tx.grid) ** 2))

    assert e_pbch > e_pss


def test_pbch_symbol_columns_are_written_in_expected_indices_normal_cp():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = rand_mib(6)
    _ = tx.generate_waveform(mib_bits=mib)

    # provjeri da su PBCH kolone za sf=0 (7..10) netrivijalne
    cols = tx._pbch_symbol_indices_for_subframe(0)
    sub = tx.grid[:, cols]
    assert np.count_nonzero(sub) > 0


def test_pbch_symbol_columns_are_written_in_expected_indices_extended_cp():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=False)
    mib = rand_mib(7)
    _ = tx.generate_waveform(mib_bits=mib)

    cols = tx._pbch_symbol_indices_for_subframe(0)
    sub = tx.grid[:, cols]
    assert np.count_nonzero(sub) > 0


def test_reserved_mask_true_everywhere_blocks_pbch_mapping_but_pss_still_present():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = rand_mib(8)

    mask = np.ones(tx.grid.shape, dtype=bool)  # sve rezervisano
    _ = tx.generate_waveform(mib_bits=mib, reserved_re_mask=mask)

    # PSS treba i dalje da bude mapiran (map_pss_to_grid ne koristi mask u tvojoj implementaciji)
    assert np.count_nonzero(tx.grid) > 0

    # ali PBCH kolone bi trebalo da budu (uglavnom) prazne
    cols = tx._pbch_symbol_indices_for_subframe(0)
    pbch_block = tx.grid[:, cols]
    assert np.count_nonzero(pbch_block) == 0


def test_reserved_mask_blocks_only_pbch_region_not_pss():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = rand_mib(9)

    mask = np.zeros(tx.grid.shape, dtype=bool)

    # blokiraj samo PBCH kolone sf0
    cols = tx._pbch_symbol_indices_for_subframe(0)
    mask[:, cols] = True

    _ = tx.generate_waveform(mib_bits=mib, reserved_re_mask=mask)

    # PSS simbol kolona treba biti nenulta
    pss_col = tx._pss_symbol_index()
    assert np.count_nonzero(tx.grid[:, pss_col]) > 0

    # PBCH sf0 kolone trebaju biti nula
    assert np.count_nonzero(tx.grid[:, cols]) == 0


# ============================================================
# 5) Determinističnost (kad nema šuma u TX)
# ============================================================

def test_same_mib_same_params_produces_same_waveform_exact():
    tx1 = LTETxChain(n_id_2=1, ndlrb=6, num_subframes=4, normal_cp=True)
    tx2 = LTETxChain(n_id_2=1, ndlrb=6, num_subframes=4, normal_cp=True)

    mib = rand_mib(123)

    w1, fs1 = tx1.generate_waveform(mib_bits=mib)
    w2, fs2 = tx2.generate_waveform(mib_bits=mib)

    assert fs1 == fs2
    assert np.allclose(w1, w2)


def test_different_mib_changes_waveform():
    tx = LTETxChain(n_id_2=1, ndlrb=6, num_subframes=4, normal_cp=True)

    w1, _ = tx.generate_waveform(mib_bits=rand_mib(1))
    w2, _ = tx.generate_waveform(mib_bits=rand_mib(2))

    assert not np.allclose(w1, w2)


def test_different_nid2_changes_waveform_even_same_mib():
    mib = rand_mib(55)

    tx0 = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    tx1 = LTETxChain(n_id_2=1, ndlrb=6, num_subframes=4, normal_cp=True)

    w0, _ = tx0.generate_waveform(mib_bits=mib)
    w1, _ = tx1.generate_waveform(mib_bits=mib)

    assert not np.allclose(w0, w1)


# ============================================================
# 6) Robustnost ulaza / tipovi
# ============================================================

def test_mib_as_python_list_accepted():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = [0, 1] * 12
    w, fs = tx.generate_waveform(mib_bits=mib)
    assert_complex_1d(w)
    assert fs > 0


def test_mib_with_values_not_just_0_1_is_masked_by_uint8_but_should_work():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = np.array([2, 3] * 12, dtype=np.uint8)  # i dalje size 24
    w, fs = tx.generate_waveform(mib_bits=mib)
    assert_complex_1d(w)
    assert fs > 0


def test_generate_waveform_returns_fs_positive_and_reasonable():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    w, fs = tx.generate_waveform()
    assert fs > 0
    # za NDLRB=6 očekuješ 1.92 MHz u tvojoj konfiguraciji
    assert abs(fs - 1.92e6) < 1e-6


# ============================================================
# 7) Edge-ish slučajevi
# ============================================================

def test_num_subframes_exactly_4_ok_for_pbch():
    tx = LTETxChain(n_id_2=2, ndlrb=6, num_subframes=4, normal_cp=True)
    w, fs = tx.generate_waveform(mib_bits=rand_mib(77))
    assert_complex_1d(w)
    assert fs > 0


def test_num_subframes_more_than_4_ok_for_pbch():
    tx = LTETxChain(n_id_2=2, ndlrb=6, num_subframes=6, normal_cp=True)
    w, fs = tx.generate_waveform(mib_bits=rand_mib(78))
    assert_complex_1d(w)
    assert fs > 0


def test_pss_only_allows_num_subframes_1():
    tx = LTETxChain(n_id_2=2, ndlrb=6, num_subframes=1, normal_cp=True)
    w, fs = tx.generate_waveform(mib_bits=None)
    assert_complex_1d(w)
    assert fs > 0


def test_grid_has_expected_num_active_subcarriers_dimension():
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    _ = tx.generate_waveform(mib_bits=rand_mib(99))
    assert tx.grid.shape[0] == 72  # 12*6
