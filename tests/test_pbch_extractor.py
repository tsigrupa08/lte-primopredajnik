"""
tests/test_pbch_extractor.py
============================

B3) Testovi za PBCHExtractor (RX)

Ovi testovi provjeravaju ispravnost ekstrakcije PBCH simbola
iz LTE resource grid-a na prijemnoj strani (RX).

Test strategija:
----------------
1) HAPPY slučajevi (10 testova):
   - Kreira se resource grid
   - PBCH simboli se mapiraju pomoću TX funkcije map_pbch_to_grid
   - PBCHExtractor na RX strani ekstrahuje simbole
   - Provjerava se da su ekstrahovani simboli IDENTIČNI
     onima koji su poslani sa predajnika (TX)

2) UNHAPPY slučajevi (10 testova):
   - Neispravne dimenzije grida
   - Pogrešan broj subcarriera
   - Grid koji nije 2D
   - NaN / Inf vrijednosti
   - Pogrešna reserved_re_mask
   - PBCH simbol indeksi van opsega

Ovim se osigurava da je RX ekstraktor:
- konzistentan sa TX mapiranjem
- robustan na pogrešne ulaze
"""

import numpy as np
import pytest

from transmitter.resource_grid import create_resource_grid, map_pbch_to_grid
from receiver.rx_resource_grid_extractor import PBCHExtractor, PBCHConfig


def rand_qpsk(n: int, seed: int) -> np.ndarray:
    """
    Generiše slučajne QPSK simbole za testiranje.

    Parametri
    ----------
    n : int
        Broj QPSK simbola.
    seed : int
        Seed za generator slučajnih brojeva (radi reproducibilnosti).

    Povratna vrijednost
    -------------------
    np.ndarray
        Niz kompleksnih QPSK simbola normalizovanih na energiju 1.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, 2))
    I = 1 - 2 * bits[:, 0]
    Q = 1 - 2 * bits[:, 1]
    return ((I + 1j * Q) / np.sqrt(2)).astype(np.complex64)


# ============================================================
# HAPPY TESTOVI (10)
# ============================================================

@pytest.mark.parametrize("seed", range(10))
def test_pbch_extractor_happy_tx_rx_match(seed):
    """
    HAPPY test:
    Provjerava da PBCHExtractor na RX strani vraća
    TAČNO iste PBCH simbole koji su mapirani na TX strani.

    Koraci:
    --------
    1) Generiše se prazan LTE resource grid
    2) Generiše se 240 QPSK PBCH simbola
    3) PBCH simboli se mapiraju u grid pomoću TX funkcije
    4) RX ekstraktor čita PBCH simbole iz grida
    5) Upoređuju se TX i RX simboli (moraju biti identični)
    """
    ndlrb = 6
    normal_cp = True
    pbch_symbol_indices = [7, 8, 9, 10]  # PBCH simboli za normal CP

    grid = create_resource_grid(
        ndlrb=ndlrb,
        num_subframes=1,
        normal_cp=normal_cp,
    )

    pbch_tx = rand_qpsk(240, seed)

    # --- TX mapiranje PBCH simbola u grid ---
    map_pbch_to_grid(
        grid=grid,
        pbch_symbols=pbch_tx,
        pbch_symbol_indices=pbch_symbol_indices,
        ndlrb=ndlrb,
    )

    # --- RX ekstrakcija PBCH simbola ---
    extractor = PBCHExtractor(
        PBCHConfig(
            ndlrb=ndlrb,
            normal_cp=normal_cp,
            pbch_symbol_indices=pbch_symbol_indices,
        )
    )
    pbch_rx = extractor.extract(grid)

    # --- Provjera ispravnosti ---
    assert pbch_rx.shape == (240,)
    assert np.allclose(pbch_rx, pbch_tx, atol=0.0, rtol=0.0)


# ============================================================
# UNHAPPY TESTOVI (10)
# ============================================================

@pytest.mark.parametrize("case", range(10))
def test_pbch_extractor_unhappy_invalid_inputs(case):
    """
    UNHAPPY testovi:
    Provjeravaju ponašanje PBCHExtractor-a za neispravne ulaze.

    Svaki 'case' predstavlja jedan tip greške ili
    rubnog (edge) slučaja koji RX mora pravilno obraditi
    (izuzetak ili degradirano ponašanje).
    """
    ndlrb = 6
    extractor = PBCHExtractor(PBCHConfig(ndlrb=ndlrb))

    if case == 0:
        # Pogrešan broj subcarriera (nije 12 * NDLRB)
        grid = np.zeros((71, 14), dtype=complex)
        with pytest.raises(AssertionError):
            extractor.extract(grid)

    elif case == 1:
        # Premalo OFDM simbola – PBCH simboli ne postoje
        grid = np.zeros((72, 13), dtype=complex)
        extractor.extract(grid)

    elif case == 2:
        # Grid nije 2D matrica
        grid = np.zeros((72,), dtype=complex)
        with pytest.raises(AssertionError):
            extractor.extract(grid)

    elif case == 3:
        # Grid realnog tipa (nije kompleksan)
        grid = np.zeros((72, 14), dtype=float)
        extractor.extract(grid)

    elif case == 4:
        # Prazan grid (nema PBCH simbola)
        grid = np.zeros((72, 14), dtype=complex)
        extractor.extract(grid)

    elif case == 5:
        # Grid sadrži NaN vrijednost
        grid = np.zeros((72, 14), dtype=complex)
        grid[0, 7] = np.nan + 1j
        with pytest.raises(ValueError):
            extractor.extract(grid)

    elif case == 6:
        # Grid sadrži Inf vrijednost
        grid = np.zeros((72, 14), dtype=complex)
        grid[0, 7] = np.inf + 1j
        with pytest.raises(ValueError):
            extractor.extract(grid)

    elif case == 7:
        # Svi RE su rezervisani – nema PBCH simbola
        grid = np.zeros((72, 14), dtype=complex)
        mask = np.ones_like(grid, dtype=bool)
        rx = extractor.extract(grid, reserved_re_mask=mask)
        assert rx.size == 0

    elif case == 8:
        # Pogrešna dimenzija reserved_re_mask
        grid = np.zeros((72, 14), dtype=complex)
        mask = np.zeros((70, 14), dtype=bool)
        with pytest.raises(IndexError):
            extractor.extract(grid, reserved_re_mask=mask)

    elif case == 9:
        # PBCH simbol indeks van opsega
        grid = np.zeros((72, 14), dtype=complex)
        extractor.cfg.pbch_symbol_indices = [20]
        with pytest.raises(ValueError):
            extractor.extract(grid)
