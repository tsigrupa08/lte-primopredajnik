import sys
import os
import numpy as np
import pytest

# Dodavanje root foldera u Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transmitter.resource_grid import create_resource_grid, map_pss_to_grid, map_pbch_to_grid

# ==============================
# HAPPY TESTS
# ==============================

def test_create_resource_grid_shape():
    """
    Provjerava dimenzije grid-a za normalni CP i 1 subframe.
    """
    grid = create_resource_grid(ndlrb=6, num_subframes=1)
    assert grid.shape == (12*6, 14)

def test_create_resource_grid_extended_cp():
    """
    Provjerava dimenzije grid-a za extended CP.
    """
    grid = create_resource_grid(ndlrb=6, num_subframes=1, normal_cp=False)
    assert grid.shape == (12*6, 12)

def test_grid_all_zero_initial():
    """
    Novi grid sadrži samo nule.
    """
    grid = create_resource_grid(ndlrb=6)
    assert np.all(grid == 0+0j)

def test_map_pss_to_grid_values():
    """
    PSS sekvenca se mapira na tačne pozicije u centru grid-a.
    """
    grid = create_resource_grid(ndlrb=6)
    pss_seq = np.exp(1j*2*np.pi*np.arange(62)/62)
    map_pss_to_grid(grid, pss_seq, symbol_index=6, ndlrb=6)
    start_idx = (6*12)//2 - 31
    np.testing.assert_array_equal(grid[start_idx:start_idx+62, 6], pss_seq)

def test_map_pss_grid_independent_positions():
    """
    Svaka vrijednost PSS sekvence se mapira na različite pozicije.
    """
    grid = create_resource_grid(ndlrb=6)
    pss_seq = np.arange(62, dtype=complex)
    map_pss_to_grid(grid, pss_seq, 6, ndlrb=6)
    start_idx = (6*12)//2 - 31
    assert grid[start_idx,6] != grid[start_idx+1,6]

def test_map_pbch_to_grid_values_no_mask():
    """
    PBCH simboli se pravilno mapiraju bez reserved mask.
    """
    grid = create_resource_grid(ndlrb=6)
    total_RE = 12*6*4
    pbch_symbols = np.arange(total_RE, dtype=complex)
    pbch_indices = [7,8,9,10]
    map_pbch_to_grid(grid, pbch_symbols, pbch_indices, ndlrb=6)
    assert grid[0,7] == pbch_symbols[0]
    assert grid[-1,10] == pbch_symbols[-1]

def test_map_pbch_to_grid_values_with_mask():
    """
    PBCH simboli se mapiraju sa reserved mask; rezervisana mjesta ostaju 0.
    """
    grid = create_resource_grid(ndlrb=6)
    pbch_symbols = np.array([1+0j]*10)
    reserved_mask = np.zeros_like(grid, dtype=bool)
    reserved_mask[0:3,7] = True
    map_pbch_to_grid(grid, pbch_symbols, [7], ndlrb=6, reserved_re_mask=reserved_mask)
    assert grid[0,7] == 0+0j
    assert grid[3,7] == 1+0j

def test_map_pbch_to_grid_exact_fit():
    """
    PBCH simboli tačno popunjavaju grid.
    """
    grid = create_resource_grid(ndlrb=6)
    total_RE = 12*6*4
    pbch_symbols = np.arange(total_RE, dtype=complex)
    pbch_indices = [7,8,9,10]
    map_pbch_to_grid(grid, pbch_symbols, pbch_indices, ndlrb=6)
    assert grid[0,7] == pbch_symbols[0]
    assert grid[-1,10] == pbch_symbols[-1]

def test_pbch_mask_all_true():
    """
    Ako je reserved mask sva True, grid ostaje prazan.
    """
    grid = create_resource_grid(ndlrb=6)
    pbch_symbols = np.ones(10, dtype=complex)
    mask = np.ones_like(grid, dtype=bool)
    map_pbch_to_grid(grid, pbch_symbols, [7], ndlrb=6, reserved_re_mask=mask)
    assert np.all(grid == 0+0j)

# ==============================
# UNHAPPY TESTS
# ==============================

def test_pss_wrong_length_raises():
    """
    PSS sekvenca dužine !=62 izaziva AssertionError.
    """
    grid = create_resource_grid(ndlrb=6)
    pss_seq = np.ones(61, dtype=complex)
    with pytest.raises(AssertionError):
        map_pss_to_grid(grid, pss_seq, 6, ndlrb=6)

def test_pss_invalid_symbol_index_raises():
    """
    PSS simbol izvan opsega izaziva AssertionError.
    """
    grid = create_resource_grid(ndlrb=6)
    pss_seq = np.ones(62, dtype=complex)
    with pytest.raises(AssertionError):
        map_pss_to_grid(grid, pss_seq, 20, ndlrb=6)

def test_pbch_symbol_index_out_of_range():
    """
    PBCH simbol sa indeksom van grid opsega izaziva AssertionError.
    """
    grid = create_resource_grid(ndlrb=6)
    pbch_symbols = np.ones(10, dtype=complex)
    with pytest.raises(AssertionError):
        map_pbch_to_grid(grid, pbch_symbols, [50], ndlrb=6)

def test_pbch_reserved_mask_wrong_shape():
    """
    PBCH reserved mask pogrešnog oblika izaziva AssertionError.
    """
    grid = create_resource_grid(ndlrb=6)
    pbch_symbols = np.ones(10, dtype=complex)
    reserved_mask = np.zeros((10,10), dtype=bool)
    with pytest.raises(AssertionError):
        map_pbch_to_grid(grid, pbch_symbols, [7], ndlrb=6, reserved_re_mask=reserved_mask)

def test_create_grid_wrong_shape_unhappy():
    """
    Grid sa pogrešnim NDLRB ne daje očekivane dimenzije.
    """
    grid = create_resource_grid(ndlrb=5)
    assert grid.shape[0] != 72  # očekivano 12*6=72

def test_pbch_fewer_symbols_than_re():
    """
    PBCH sa manje simbola nego RE mjesta ne popunjava cijeli grid.
    """
    grid = create_resource_grid(ndlrb=6)
    pbch_symbols = np.ones(10, dtype=complex)
    map_pbch_to_grid(grid, pbch_symbols, [7], ndlrb=6)
    assert grid[10,7] == 0+0j

def test_pbch_more_symbols_than_re():
    """
    PBCH sa više simbola nego što grid može primiti – višak se ignorira.
    """
    grid = create_resource_grid(ndlrb=6)
    pbch_symbols = np.arange(1000, dtype=complex)
    map_pbch_to_grid(grid, pbch_symbols, [7,8,9,10], ndlrb=6)
    assert grid[-1,10] == pbch_symbols[287]  # 12*6*4=288 RE

def test_pbch_non_complex_symbols():
    """
    PBCH simboli koji nisu kompleksni se konvertuju u complex.
    """
    grid = create_resource_grid(ndlrb=6)
    pbch_symbols = np.arange(10)  # int simboli
    map_pbch_to_grid(grid, pbch_symbols, [7], ndlrb=6)
    assert np.iscomplexobj(grid[0:10,7])
