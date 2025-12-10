import os
import sys
import numpy as np
import pytest

# -------------------------------------------------------------------
# Omogućavanje importa 'transmitter' paketa
# -------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transmitter.resource_grid import (
    create_resource_grid,
    map_pss_to_grid,
    map_pbch_to_grid,
)


# ===================================================================
#                           HAPPY TESTS
# ===================================================================

def test_create_resource_grid_shape_normal_cp():
    grid = create_resource_grid(ndlrb=6, num_subframes=1, normal_cp=True)
    assert grid.shape == (72, 14)


def test_create_resource_grid_shape_extended_cp():
    grid = create_resource_grid(ndlrb=6, num_subframes=1, normal_cp=False)
    assert grid.shape == (72, 12)


def test_grid_initial_zero():
    grid = create_resource_grid(ndlrb=6)
    assert np.all(grid == 0)


def test_pss_mapping_correct_positions():
    ndlrb = 6
    grid = create_resource_grid(ndlrb)
    pss = np.exp(1j * 2 * np.pi * np.arange(62) / 62)
    symbol_index = 6

    map_pss_to_grid(grid, pss, symbol_index, ndlrb=ndlrb)

    start = (12 * ndlrb) // 2 - 31
    for i in range(62):
        assert grid[start + i, symbol_index] == pss[i]


def test_pbch_maps_sequentially_without_mask():
    grid = create_resource_grid(ndlrb=6)
    pbch_syms = np.ones(240, dtype=complex)
    indices = [7, 8, 9, 10]

    map_pbch_to_grid(grid, pbch_syms, indices, ndlrb=6)

    # PBCH počinje popunjavati od (0,7) pa dalje dole po subcarrierima
    assert grid[0, 7] == 1 + 0j
    assert grid[1, 7] == 1 + 0j
    assert grid[71, 7] == 1 + 0j             # zadnji subcarrier u simbolu 7
    assert grid[0, 8] == 1 + 0j              # prelazi u sljedeći simbol


def test_pbch_stops_when_exhausted():
    grid = create_resource_grid(ndlrb=6)
    pbch_syms = np.ones(10, dtype=complex)
    indices = [7, 8]

    map_pbch_to_grid(grid, pbch_syms, indices, ndlrb=6)

    # Samo prvih 10 RE treba biti popunjeno
    filled = np.sum(grid != 0)
    assert filled == 10


def test_pbch_skips_reserved_mask():
    grid = create_resource_grid(ndlrb=6)
    pbch = np.arange(20, dtype=complex)

    mask = np.zeros_like(grid, dtype=bool)
    mask[0:5, 7] = True     # rezerviši prvih 5 RE u simbolu 7

    map_pbch_to_grid(grid, pbch, [7], ndlrb=6, reserved_re_mask=mask)

    # Prvih 5 RE NE smiju biti popunjeni
    assert np.all(grid[0:5, 7] == 0)

    # Prvi PBCH simbol treba otići na grid[5,7]
    assert grid[5, 7] == 0 + 0j or grid[5, 7] == pbch[0]  # validno u ovisnosti o početku
    assert grid[6, 7] == pbch[1]


# ===================================================================
#                           UNHAPPY TESTS
# ===================================================================

def test_pss_fails_wrong_length():
    grid = create_resource_grid(ndlrb=6)
    pss_wrong = np.ones(61, dtype=complex)

    with pytest.raises(AssertionError):
        map_pss_to_grid(grid, pss_wrong, 6, ndlrb=6)


def test_pss_symbol_index_out_of_range():
    grid = create_resource_grid(ndlrb=6)
    pss = np.ones(62, dtype=complex)

    with pytest.raises(AssertionError):
        map_pss_to_grid(grid, pss, 100, ndlrb=6)


def test_pss_fails_grid_ndlrb_inconsistent():
    grid = np.zeros((80, 14), dtype=complex)  # pogrešne dimenzije
    pss = np.ones(62)

    with pytest.raises(AssertionError):
        map_pss_to_grid(grid, pss, 6, ndlrb=6)


def test_pbch_fails_symbol_index_out_of_range():
    grid = create_resource_grid(ndlrb=6)
    pbch = np.ones(10)

    with pytest.raises(AssertionError):
        map_pbch_to_grid(grid, pbch, [999], ndlrb=6)


def test_pbch_fails_reserved_mask_wrong_shape():
    grid = create_resource_grid(ndlrb=6)
    pbch = np.ones(20)
    mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(AssertionError):
        map_pbch_to_grid(grid, pbch, [7], ndlrb=6, reserved_re_mask=mask)


def test_pbch_fails_grid_ndlrb_mismatch():
    grid = np.zeros((80, 14), dtype=complex)
    pbch = np.ones(20)

    with pytest.raises(AssertionError):
        map_pbch_to_grid(grid, pbch, [7], ndlrb=6)


def test_create_resource_grid_multiple_subframes():
    grid = create_resource_grid(ndlrb=6, num_subframes=3)
    assert grid.shape == (72, 42)  # 3 × 14 simbola


def test_pbch_maps_into_multiple_symbols():
    grid = create_resource_grid(ndlrb=6)
    pbch = np.ones(200)
    map_pbch_to_grid(grid, pbch, [7, 8, 9], ndlrb=6)

    assert np.sum(grid != 0) == 200


def test_pss_does_not_overwrite_other_symbols():
    grid = create_resource_grid(ndlrb=6)
    grid[:, 5] = 5 + 1j   # neka druga modulacija

    pss = np.ones(62, dtype=complex)
    map_pss_to_grid(grid, pss, 6, ndlrb=6)

    # Simbol 5 ne smije biti promijenjen
    assert np.all(grid[:, 5] == 5 + 1j)


def test_pbch_does_not_write_outside_indices():
    grid = create_resource_grid(ndlrb=6)
    pbch = np.ones(20)

    map_pbch_to_grid(grid, pbch, [10], ndlrb=6)

    # Samo simbol 10 smije imati podatke
    assert np.sum(grid[:, :10] != 0) == 0
    assert np.sum(grid[:, 11:] != 0) == 0

