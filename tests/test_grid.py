import sys
import os
import numpy as np
import pytest

# Dodavanje root foldera u Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import klase iz modula resource_grid.py
from transmitter.resource_grid import ResourceGrid

# =============================
# UNIT TESTOVI ZA ResourceGrid
# =============================

# -----------------------------
# HAPPY TESTS
# -----------------------------

def test_create_resource_grid_shape():
    """
    Testira dimenzije kreiranog resource grid-a.

    Provjerava da li `ResourceGrid` vraća grid sa ispravnim brojem redova i kolona
    za normalni CP.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6, num_subframes=1)
    assert obj.grid.shape == (12 * obj.ndlrb, 14), "Grid dimenzije nisu ispravne za normalni CP"


def test_map_pss_to_grid_values():
    """
    Testira ispravno mapiranje PSS sekvence u centar grid-a.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    pss_seq = np.exp(1j * 2 * np.pi * np.arange(62) / 62)
    symbol_index = 6
    obj.map_pss(pss_seq, symbol_index)
    start_idx = (obj.ndlrb * 12) // 2 - 31
    for n in range(62):
        assert obj.grid[start_idx + n, symbol_index] == pss_seq[n]


def test_map_pbch_to_grid_values_no_mask():
    """
    Testira PBCH mapiranje bez reserved mask.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    total_RE = 12*obj.ndlrb*4
    pbch_symbols = np.arange(total_RE, dtype=complex)
    pbch_indices = [7,8,9,10]
    obj.map_pbch(pbch_symbols, pbch_indices)
    assert obj.grid[0,7] == pbch_symbols[0]
    assert obj.grid[-1,10] == pbch_symbols[-1]


def test_map_pbch_to_grid_values_with_mask():
    """
    Testira PBCH mapiranje sa reserved mask.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    pbch_symbols = np.array([1+0j]*10)
    pbch_indices = [7]
    reserved_mask = np.zeros_like(obj.grid, dtype=bool)
    reserved_mask[0:3,7] = True
    obj.map_pbch(pbch_symbols, pbch_indices, reserved_re_mask=reserved_mask)
    assert obj.grid[0,7] == 0+0j
    assert obj.grid[3,7] == 1+0j


def test_map_pbch_to_grid_exact_fit():
    """
    Testira PBCH mapiranje gdje simboli tačno popunjavaju grid.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    total_RE = 12*obj.ndlrb*4
    pbch_symbols = np.arange(total_RE, dtype=complex)
    pbch_indices = [7,8,9,10]
    obj.map_pbch(pbch_symbols, pbch_indices)
    assert obj.grid[0,7] == pbch_symbols[0]
    assert obj.grid[0,10] == pbch_symbols[12*obj.ndlrb*3]
    assert obj.grid[-1,10] == pbch_symbols[-1]


def test_map_pss_grid_independent_positions():
    """
    Testira da PSS mapira različite vrijednosti na različite pozicije.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    pss_seq = np.arange(62, dtype=complex)
    obj.map_pss(pss_seq, 6)
    start_idx = (obj.ndlrb*12)//2 - 31
    assert obj.grid[start_idx,6] != obj.grid[start_idx+1,6]


def test_grid_all_zero_initial():
    """
    Provjerava da novi grid inicijalno sadrži samo nule.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    assert np.all(obj.grid == 0+0j)


def test_pbch_mask_all_true():
    """
    Provjerava da grid ostaje nepopunjen ako je reserved mask sva True.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    pbch_symbols = np.ones(10, dtype=complex)
    mask = np.ones_like(obj.grid, dtype=bool)
    obj.map_pbch(pbch_symbols, [7], reserved_re_mask=mask)
    assert np.all(obj.grid == 0+0j)


# -----------------------------
# UNHAPPY TESTS
# -----------------------------

def test_pss_wrong_length_raises():
    """
    Testira grešku kada PSS sekvenca nije dužine 62.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Ako je sekvenca PSS dužine različite od 62.
    """
    obj = ResourceGrid(ndlrb=6)
    pss_seq = np.ones(61, dtype=complex)
    with pytest.raises(AssertionError):
        obj.map_pss(pss_seq, 6)


def test_pss_invalid_symbol_index_raises():
    """
    Testira grešku kada je OFDM simbol izvan opsega.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Ako je simbol izvan validnog opsega.
    """
    obj = ResourceGrid(ndlrb=6)
    pss_seq = np.ones(62, dtype=complex)
    with pytest.raises(AssertionError):
        obj.map_pss(pss_seq, 20)


def test_pbch_symbol_index_out_of_range():
    """
    Testira PBCH simbol sa indeksom van grid opsega.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Ako je PBCH indeks izvan granica grid-a.
    """
    obj = ResourceGrid(ndlrb=6)
    pbch_symbols = np.ones(10, dtype=complex)
    pbch_indices = [50]
    with pytest.raises(AssertionError):
        obj.map_pbch(pbch_symbols, pbch_indices)


def test_pbch_reserved_mask_wrong_shape():
    """
    Testira PBCH reserved mask sa pogrešnim dimenzijama.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Ako reserved mask nema ispravnu dimenziju.
    """
    obj = ResourceGrid(ndlrb=6)
    pbch_symbols = np.ones(10, dtype=complex)
    reserved_mask = np.zeros((10,10), dtype=bool)
    with pytest.raises(AssertionError):
        obj.map_pbch(pbch_symbols, [7], reserved_re_mask=reserved_mask)


def test_create_grid_wrong_shape_unhappy():
    """
    Testira kreiranje grid-a sa pogrešnim NDLRB.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=5)
    assert obj.grid.shape[0] != 72  # 12*6 =72, sad 12*5=60


def test_pbch_fewer_symbols_than_re():
    """
    Testira PBCH sa manje simbola nego što ima grid-ova mjesta.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    pbch_symbols = np.ones(10, dtype=complex)
    obj.map_pbch(pbch_symbols, [7])
    assert obj.grid[10,7] == 0+0j


def test_pbch_more_symbols_than_re():
    """
    Testira PBCH sa više simbola nego što grid može primiti.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    obj = ResourceGrid(ndlrb=6)
    pbch_symbols = np.arange(1000, dtype=complex)
    pbch_indices = [7,8,9,10]
    obj.map_pbch(pbch_symbols, pbch_indices)
    assert obj.grid[-1,10] == pbch_symbols[287]
