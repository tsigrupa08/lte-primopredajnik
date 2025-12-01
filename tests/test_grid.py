import sys
import os
import numpy as np
import pytest

# -----------------------------
# Dodavanje root foldera u Python path
# Ovo omogućava Python-u da pronađe paket 'transmitter' unutar projekta
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import funkcija iz modula resource_grid.py
from transmitter.resource_grid import create_resource_grid, map_pss_to_grid, map_pbch_to_grid

# =============================
# Unit testovi za resource_grid.py
# =============================

def test_create_resource_grid_shape():
    """
    Testira funkciju create_resource_grid.
    Provjerava da li kreirani grid ima ispravne dimenzije.
    
    Za normalni CP i ndlrb=6:
    - broj podnosioca = 12*6 = 72
    - broj OFDM simbola po subfrejmu = 14
    """
    ndlrb = 6
    num_subframes = 1
    grid = create_resource_grid(ndlrb=ndlrb, num_subframes=num_subframes)

    # Provjera da dimenzije odgovaraju očekivanom obliku (72,14)
    assert grid.shape == (12 * ndlrb, 14), "Grid dimenzije nisu ispravne za normalni CP"


def test_map_pss_to_grid_values():
    """
    Testira funkciju map_pss_to_grid.
    Provjerava da PSS sekvenca bude pravilno mapirana u centar grid-a.
    
    Koraci:
    1. Kreira se dummy PSS sekvenca (62 kompleksna broja)
    2. Mapira se na zadani OFDM simbol (l=6)
    3. Provjerava se da su elementi u grid-u na očekivanim pozicijama jednaki PSS sekvenci
    """
    ndlrb = 6
    grid = create_resource_grid(ndlrb=ndlrb)
    pss_seq = np.exp(1j * 2 * np.pi * np.arange(62) / 62)  # dummy PSS sekvenca
    symbol_index = 6  # zadani OFDM simbol
    map_pss_to_grid(grid, pss_seq, symbol_index, ndlrb=ndlrb)

    # Indeks početka PSS sekvence u gridu
    start_idx = (ndlrb * 12) // 2 - 31
    for n in range(62):
        # Provjera da su PSS simboli na očekivanim pozicijama
        assert grid[start_idx + n, symbol_index] == pss_seq[n], f"PSS simbol na poziciji {start_idx+n} nije ispravan"


def test_map_pbch_to_grid_values_no_mask():
    """
    Testira funkciju map_pbch_to_grid bez reserved RE mask.
    
    Koraci:
    1. Kreira se grid
    2. Dummy PBCH simboli (240) mapiraju se na OFDM simbole l=7,8,9,10
    3. Provjerava se:
       - prvi PBCH simbol u prvom OFDM simbolu
       - zadnji PBCH simbol u posljednjem OFDM simbolu
    """
    ndlrb = 6
    grid = create_resource_grid(ndlrb=ndlrb)
    pbch_symbols = np.array([1+1j]*240)  # dummy PBCH simboli
    pbch_indices = [7,8,9,10]  # OFDM simboli za PBCH
    map_pbch_to_grid(grid, pbch_symbols, pbch_indices, ndlrb=ndlrb)

    # Provjera prvog PBCH simbola
    assert grid[0,7] == 1+1j, "PBCH simboli nisu mapirani pravilno"
    # Provjera zadnjeg PBCH simbola
    assert grid[239 % (12*ndlrb), pbch_indices[-1]] == 1+1j, "Zadnji PBCH simbol nije mapiran pravilno"


def test_map_pbch_to_grid_values_with_mask():
    """
    Testira funkciju map_pbch_to_grid sa reserved RE mask.
    
    Koraci:
    1. Kreira se grid
    2. Dummy PBCH simboli (10) mapiraju se na OFDM simbol l=7
    3. Postavlja se reserved RE maska na prva 3 RE
    4. Provjerava se:
       - PBCH simboli preskaču rezervisana mjesta
       - prvi PBCH simbol nakon maskiranih RE je na indexu 3
    """
    ndlrb = 6
    grid = create_resource_grid(ndlrb=ndlrb)
    pbch_symbols = np.array([1+0j]*10)  # dummy PBCH simboli
    pbch_indices = [7]  # samo jedan OFDM simbol
    reserved_mask = np.zeros_like(grid, dtype=bool)
    reserved_mask[0:3,7] = True  # prvi 3 RE su rezervisana

    map_pbch_to_grid(grid, pbch_symbols, pbch_indices, ndlrb=ndlrb, reserved_re_mask=reserved_mask)

    # Provjera da PBCH preskače rezervisana mjesta
    assert grid[0,7] == 0+0j
    assert grid[1,7] == 0+0j
    assert grid[2,7] == 0+0j
    # Prvi PBCH simbol bi trebao biti na indexu 3
    assert grid[3,7] == 1+0j, "PBCH simbol nije preskočio rezervisane RE"
