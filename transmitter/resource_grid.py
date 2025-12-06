"""
resource_grid.py

Modul za kreiranje LTE resource grid-a i mapiranje PSS i PBCH simbola
u centar 6 RB (NDLRB = 6), u skladu sa poglavljima 7.5 i 7.6
knjige "Digital Signal Processing in Modern Communication Systems"
(A. Schwarzinger).

Ovaj modul koristi OOP pristup preko klase ResourceGrid, ali sadrži i
wrapper funkcije radi kompatibilnosti sa starim pozivima.
"""

from typing import Iterable, Optional

import numpy as np


class ResourceGrid:
    """
    Klasa za reprezentaciju LTE resource grid-a i mapiranje PSS i PBCH simbola.

    Resource grid je dvodimenzionalna matrica kompleksnih brojeva dimenzija
    (12 * NDLRB, broj_OFDM_simbola), gdje redovi predstavljaju podnosioca
    (subcarriere), a kolone OFDM simbole.

    Parametri
    ----------
    ndlrb : int, opcionalno
        Broj downlink resource blokova (NDLRB). Za 1.4 MHz LTE vrijedi 6.
    num_subframes : int, opcionalno
        Broj LTE subfrejmova u gridu.
    normal_cp : bool, opcionalno
        Ako je True, koristi se normalni CP (14 simbola po subfrejmu),
        u suprotnom prošireni CP (12 simbola po subfrejmu).

    Atributi
    --------
    ndlrb : int
        Broj downlink resource blokova.
    num_subframes : int
        Broj subfrejmova u gridu.
    normal_cp : bool
        Označava da li se koristi normalni ili prošireni CP.
    num_subcarriers : int
        Ukupan broj podnosioca (12 * ndlrb).
    num_symbols_per_subframe : int
        Broj OFDM simbola po subfrejmu (14 ili 12).
    num_symbols_total : int
        Ukupan broj OFDM simbola u gridu.
    grid : np.ndarray (complex128)
        Kompleksna matrica oblika (num_subcarriers, num_symbols_total)
        koja predstavlja LTE resource grid.

    Primjer
    -------
    >>> from transmitter.resource_grid import ResourceGrid
    >>> rg = ResourceGrid(ndlrb=6, num_subframes=1, normal_cp=True)
    >>> rg.grid.shape
    (72, 14)
    """

    def __init__(
        self,
        ndlrb: int = 6,
        num_subframes: int = 1,
        normal_cp: bool = True,
    ) -> None:
        self.ndlrb = ndlrb
        self.num_subframes = num_subframes
        self.normal_cp = normal_cp

        self.num_subcarriers = 12 * ndlrb
        self.num_symbols_per_subframe = 14 if normal_cp else 12
        self.num_symbols_total = self.num_symbols_per_subframe * num_subframes

        self.grid = np.zeros(
            (self.num_subcarriers, self.num_symbols_total),
            dtype=complex,
        )

    def map_pss(
        self,
        pss_sequence: np.ndarray,
        symbol_index: int,
    ) -> None:
        """
        Mapira PSS sekvencu u centar resource grid-a na zadati OFDM simbol.

        PSS sekvenca dužine 62 se mapira na 62 uzastopna resource elementa
        (RE) oko DC komponente, u skladu sa specifikacijom za NDLRB = 6.
        Za normalni CP i subfrejm 0, PSS je u zadnjem simbolu slota 0 (l = 6).

        Parametri
        ----------
        pss_sequence : np.ndarray (complex128)
            Kompleksni niz dužine 62 koji predstavlja PSS sekvencu
            u frekvencijskom domenu (Zadoff–Chu).
        symbol_index : int
            Indeks OFDM simbola (kolona) u koji se PSS mapira.
            Mora biti u opsegu [0, broj_OFDM_simbola - 1].

        Povratna vrijednost
        -------------------
        None
            Funkcija radi in-place i mijenja atribut `grid` unutar objekta.

        Izuzeci
        -------
        AssertionError
            Ako dimenzije grida ne odgovaraju 12 * NDLRB.
            Ako PSS sekvenca nema tačno 62 elementa.
            Ako je symbol_index van važećeg opsega.

        Primjer
        -------
        >>> from transmitter.resource_grid import ResourceGrid
        >>> import numpy as np
        >>> rg = ResourceGrid(ndlrb=6)
        >>> pss = np.ones(62, dtype=complex)
        >>> rg.map_pss(pss_sequence=pss, symbol_index=6)
        >>> rg.grid[0, 6]  # jedan od mapiranih RE
        (1+0j)
        """
        num_subcarriers, num_symbols_total = self.grid.shape

        assert num_subcarriers == 12 * self.ndlrb, "Grid i NDLRB nisu konzistentni."
        assert pss_sequence.shape[0] == 62, "PSS sekvenca mora imati tačno 62 elementa."
        assert 0 <= symbol_index < num_symbols_total, "symbol_index je van opsega."

        center = (self.ndlrb * 12) // 2  # npr. za NDLRB=6 → 72/2 = 36
        k0 = center - 31                  # početni indeks u gridu

        for n in range(62):
            k = k0 + n
            self.grid[k, symbol_index] = pss_sequence[n]

    def map_pbch(
        self,
        pbch_symbols: np.ndarray,
        pbch_symbol_indices: Iterable[int],
        reserved_re_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Mapira PBCH QPSK simbole u resource grid (centar 6 RB).

        PBCH se mapira na zadane OFDM simbole (tipično l = 7, 8, 9, 10
        u subfrejmu 0 za normalni CP), u centralnih 6 resource blokova
        (NDLRB = 6). Ako je zadana `reserved_re_mask`, PBCH preskače te RE.

        Parametri
        ----------
        pbch_symbols : np.ndarray (complex128)
            Jednodimenzionalni niz kompleksnih QPSK simbola (npr. 240 simbola
            za normalni CP nakon CRC, konvolucijskog kodiranja i
            rate matchinga).
        pbch_symbol_indices : Iterable[int]
            Indeksi OFDM simbola (kolone u gridu) u koje se PBCH mapira.
        reserved_re_mask : np.ndarray (bool), opcionalno
            Bool matrica istih dimenzija kao `grid`, gdje je True na
            pozicijama koje su rezervisane (CRS itd.). PBCH se ne mapira
            na te resource elemente.

        Povratna vrijednost
        -------------------
        None
            Funkcija radi in-place i mijenja atribut `grid` unutar objekta.

        Izuzeci
        -------
        AssertionError
            Ako dimenzije grida ne odgovaraju 12 * NDLRB,
            ako je neki indeks u pbch_symbol_indices van opsega,
            ili ako reserved_re_mask nije istih dimenzija kao grid.

        Primjer
        -------
        >>> from transmitter.resource_grid import ResourceGrid
        >>> import numpy as np
        >>> rg = ResourceGrid(ndlrb=6)
        >>> pbch = np.ones(240, dtype=complex)
        >>> rg.map_pbch(pbch_symbols=pbch, pbch_symbol_indices=[7, 8, 9, 10])
        """
        num_subcarriers, num_symbols_total = self.grid.shape
        num_subcarriers_expected = 12 * self.ndlrb
        assert num_subcarriers == num_subcarriers_expected, "Grid i NDLRB nisu konzistentni."

        pbch_symbol_indices = list(pbch_symbol_indices)
        for l in pbch_symbol_indices:
            assert 0 <= l < num_symbols_total, "PBCH simbol indeks je van opsega."

        if reserved_re_mask is not None:
            assert reserved_re_mask.shape == self.grid.shape, (
                "reserved_re_mask mora imati iste dimenzije kao grid."
            )

        max_k = num_subcarriers_expected - 1
        symbol_ptr = 0  # indeks u pbch_symbols

        for l in pbch_symbol_indices:
            k = 0
            while k <= max_k and symbol_ptr < pbch_symbols.shape[0]:
                if reserved_re_mask is not None and reserved_re_mask[k, l]:
                    # CRS ili neki drugi rezervisani RE – preskačemo
                    k += 1
                    continue

                self.grid[k, l] = pbch_symbols[symbol_ptr]
                symbol_ptr += 1
                k += 1

            if symbol_ptr >= pbch_symbols.shape[0]:
                break  # sve PBCH simbole smo iskoristili


# -------------------------------------------------------------------------
# Pomoćne/wrapper funkcije radi kompatibilnosti sa starim kodom i testovima
# -------------------------------------------------------------------------


def create_resource_grid(
    ndlrb: int = 6,
    num_subframes: int = 1,
    normal_cp: bool = True,
) -> np.ndarray:
    """
    Kreira prazan LTE resource grid (kompatibilno sa starom verzijom).

    Parametri
    ----------
    ndlrb : int, opcionalno
        Broj downlink resource blokova (NDLRB).
    num_subframes : int, opcionalno
        Broj subfrejmova u gridu.
    normal_cp : bool, opcionalno
        Ako je True, koristi se normalni CP (14 simbola po subfrejmu),
        u suprotnom prošireni CP (12 simbola po subfrejmu).

    Povratna vrijednost
    -------------------
    np.ndarray (complex128)
        Prazan resource grid dimenzija (12 * ndlrb, broj_OFDM_simbola).

    Primjer
    -------
    >>> from transmitter.resource_grid import create_resource_grid
    >>> grid = create_resource_grid(ndlrb=6, num_subframes=1, normal_cp=True)
    >>> grid.shape
    (72, 14)
    """
    rg = ResourceGrid(ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=normal_cp)
    return rg.grid


def _infer_grid_params(grid: np.ndarray, ndlrb: int) -> tuple[int, bool, int]:
    """
    Interna pomoćna funkcija: iz oblika grida procjenjuje CP i broj subfrejmova.

    Parametri
    ----------
    grid : np.ndarray
        Ulazni resource grid.
    ndlrb : int
        Broj downlink resource blokova.

    Povratna vrijednost
    -------------------
    tuple
        Tuple (ndlrb, normal_cp, num_subframes).

    Izuzeci
    -------
    AssertionError
        Ako broj podnosioca ne odgovara 12 * ndlrb.
    ValueError
        Ako se na osnovu broja simbola ne može odrediti CP (14 ili 12).
    """
    num_subcarriers, num_symbols_total = grid.shape
    assert num_subcarriers == 12 * ndlrb, "Grid i NDLRB nisu konzistentni."

    if num_symbols_total % 14 == 0:
        normal_cp = True
        num_subframes = num_symbols_total // 14
    elif num_symbols_total % 12 == 0:
        normal_cp = False
        num_subframes = num_symbols_total // 12
    else:
        raise ValueError("Ne mogu da odredim CP i broj subfrejmova iz oblika grida.")

    return ndlrb, normal_cp, num_subframes


def map_pss_to_grid(
    grid: np.ndarray,
    pss_sequence: np.ndarray,
    symbol_index: int,
    ndlrb: int = 6,
) -> None:
    """
    Wrapper funkcija za mapiranje PSS-a (radi kompatibilnosti sa starim kodom).

    Interno kreira ResourceGrid objekat, postavlja njegov `grid` na ulazni
    i poziva `ResourceGrid.map_pss`.

    Parametri
    ----------
    grid : np.ndarray (complex128)
        Resource grid dimenzija (12 * ndlrb, broj_OFDM_simbola).
    pss_sequence : np.ndarray (complex128)
        PSS sekvenca dužine 62.
    symbol_index : int
        Indeks OFDM simbola u koji se PSS mapira.
    ndlrb : int, opcionalno
        Broj downlink resource blokova.

    Povratna vrijednost
    -------------------
    None
        Funkcija radi in-place i mijenja sadržaj `grid`.

    Primjer
    -------
    >>> from transmitter.resource_grid import create_resource_grid, map_pss_to_grid
    >>> import numpy as np
    >>> grid = create_resource_grid(ndlrb=6)
    >>> pss = np.ones(62, dtype=complex)
    >>> map_pss_to_grid(grid, pss, symbol_index=6, ndlrb=6)
    """
    ndlrb, normal_cp, num_subframes = _infer_grid_params(grid, ndlrb)
    rg = ResourceGrid(ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=normal_cp)
    rg.grid = grid
    rg.map_pss(pss_sequence, symbol_index)


def map_pbch_to_grid(
    grid: np.ndarray,
    pbch_symbols: np.ndarray,
    pbch_symbol_indices: Iterable[int],
    ndlrb: int = 6,
    reserved_re_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Wrapper funkcija za mapiranje PBCH simbola (radi kompatibilnosti).

    Interno kreira ResourceGrid objekat, postavlja njegov `grid` na ulazni
    i poziva `ResourceGrid.map_pbch`.

    Parametri
    ----------
    grid : np.ndarray (complex128)
        Resource grid dimenzija (12 * ndlrb, broj_OFDM_simbola).
    pbch_symbols : np.ndarray (complex128)
        Jednodimenzionalni niz QPSK simbola.
    pbch_symbol_indices : Iterable[int]
        Indeksi OFDM simbola u koje se PBCH mapira.
    ndlrb : int, opcionalno
        Broj downlink resource blokova.
    reserved_re_mask : np.ndarray (bool), opcionalno
        Maska rezervisanih RE (isti oblik kao grid).

    Povratna vrijednost
    -------------------
    None
        Funkcija radi in-place i mijenja sadržaj `grid`.

    Primjer
    -------
    >>> from transmitter.resource_grid import create_resource_grid, map_pbch_to_grid
    >>> import numpy as np
    >>> grid = create_resource_grid(ndlrb=6)
    >>> pbch = np.ones(240, dtype=complex)
    >>> map_pbch_to_grid(grid, pbch, pbch_symbol_indices=[7, 8, 9, 10], ndlrb=6)
    """
    ndlrb, normal_cp, num_subframes = _infer_grid_params(grid, ndlrb)
    rg = ResourceGrid(ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=normal_cp)
    rg.grid = grid
    rg.map_pbch(pbch_symbols, pbch_symbol_indices, reserved_re_mask=reserved_re_mask)

  
