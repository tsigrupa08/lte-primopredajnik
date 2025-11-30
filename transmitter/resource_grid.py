"""
resource_grid.py

Modul za kreiranje LTE resource grid-a i mapiranje PSS i PBCH simbola
u centar 6 RB (NDLRB = 6), u skladu sa poglavljima 7.5 i 7.6
knjige "Digital Signal Processing in Modern Communication Systems"
(A.. Schwarzinger).

"""

from typing import Iterable, Optional

import numpy as np


def create_resource_grid(
    ndlrb: int = 6,
    num_subframes: int = 1,
    normal_cp: bool = True,
) -> np.ndarray:
    """
    Kreira prazan LTE resource grid.

    Grid ima dimenzije:
        broj_podnosioca = 12 * ndlrb
        broj_OFDM_simbola = (14 ili 12) * num_subframes

    Prema knjizi, resource grid ima 12 * NDLRB podnosiocaa (DC se ne broji) i
    14 ofdm simbola po subfrejmu za normalni CP, odnosno 12 za prošireni CP.
    """
    num_subcarriers = 12 * ndlrb
    num_symbols_per_subframe = 14 if normal_cp else 12
    num_symbols_total = num_symbols_per_subframe * num_subframes

    grid = np.zeros((num_subcarriers, num_symbols_total), dtype=complex)
    return grid


def map_pss_to_grid(
    grid: np.ndarray,
    pss_sequence: np.ndarray,
    symbol_index: int,
    ndlrb: int = 6,
) -> None:
    """
    Mapira PSS sekvencu u centar resource grid-a na zadati OFDM simbol.

    Parametri:
    - grid: resource grid dimenzija (12*NDLRB, broj_OFDM_simbola)
    - pss_sequence: kompleksni niz dužine 62 (Zadoff–Chu u frekv. domenu)
    - symbol_index: indeks OFDM simbola (kolona) u koji se PSS mapira
      Za normalni CP i subfrejm 0, PSS je u zadnjem simbolu slota 0:
        slot 0 → simboli 0..6 → PSS na l = 6
    - ndlrb: broj downlink resource blokova (za 1.4 MHz je 6)

     PSS se mapira u 62 RE u centru grida, sa indeksima:
        k = (NDLRB*12)/2 - 31 + n,  n = 0..61
    Za NDLRB = 6 → broj_podnosioca = 72, k = 5..66.
    """
    num_subcarriers, num_symbols_total = grid.shape

    # Provjere da se lakše uhvate bagovi
    assert num_subcarriers == 12 * ndlrb, "Grid i NDLRB nisu konzistentni."
    assert pss_sequence.shape[0] == 62, "PSS sekvenca mora imati tačno 62 elementa."
    assert 0 <= symbol_index < num_symbols_total, "symbol_index je van opsega."

    center = (ndlrb * 12) // 2  # npr. za NDLRB=6 → 72/2 = 36
    k0 = center - 31            # početni indeks u gridu

    for n in range(62):
        k = k0 + n
        grid[k, symbol_index] = pss_sequence[n]


def map_pbch_to_grid(
    grid: np.ndarray,
    pbch_symbols: np.ndarray,
    pbch_symbol_indices: Iterable[int],
    ndlrb: int = 6,
    reserved_re_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Mapira PBCH QPSK simbole u resource grid, u centar 6 RB.

    Parametri:
    - grid: resource grid dimenzija (12*NDLRB, broj_OFDM_simbola)
    - pbch_symbols: 1D niz kompleksnih brojeva (npr. 240 simbola za normalni CP)
      To su već QPSK simboli nakon cijelog PBCH kodiranja (CRC, TBC, rate matching, scrambling).
    - pbch_symbol_indices: iterabilan skup indeksa OFDM simbola koji pripadaju PBCH-u.
      PBCH zauzima prva 4 OFDM simbola drugog slota subfrejma 0.
      Za normalni CP:
        subfrejm ima 14 simbola, slot 0: l=0..6, slot 1: l=7..13
        PBCH je na l = 7, 8, 9, 10
    - ndlrb: broj downlink resource blokova (6 za 1.4 MHz)
    - reserved_re_mask: opciona bool matrica iste dimenzije kao grid, gdje je True
      na pozicijama koje su zauzete (npr. CRS). PBCH se NE mapira na te RE.

     pravila mapiranja su:
      * prvi kompleksni broj ide na najniži raspoloživi k u prvom PBCH simbolu,
      * zatim se k povećava za 1,
      * kad dođemo do k = Max = NDLRB*12 - 1, prelazimo na k=0 sljedećeg PBCH simbola,
      * resource elementi rezervisani za CRS moraju se preskočiti.
    """
    num_subcarriers, num_symbols_total = grid.shape
    num_subcarriers_expected = 12 * ndlrb
    assert num_subcarriers == num_subcarriers_expected, "Grid i NDLRB nisu konzistentni."

    pbch_symbol_indices = list(pbch_symbol_indices)
    for l in pbch_symbol_indices:
        assert 0 <= l < num_symbols_total, "PBCH simbol indeks je van opsega."

    if reserved_re_mask is not None:
        assert reserved_re_mask.shape == grid.shape, "reserved_re_mask mora imati iste dimenzije kao grid."

    max_k = num_subcarriers_expected - 1
    symbol_ptr = 0  # indeks u pbch_symbols

    
    for l in pbch_symbol_indices:
        k = 0
        while k <= max_k and symbol_ptr < pbch_symbols.shape[0]:
            if reserved_re_mask is not None and reserved_re_mask[k, l]:
                # CRS ili neki drugi rezervisani RE – preskačemo
                k += 1
                continue

            grid[k, l] = pbch_symbols[symbol_ptr]
            symbol_ptr += 1
            k += 1

        if symbol_ptr >= pbch_symbols.shape[0]:
            break  # sve PBCH simbole smo iskoristili

  
