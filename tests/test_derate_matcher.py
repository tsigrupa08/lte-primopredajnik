"""
tests/test_derate_matcher.py
============================

Testovi za DeRateMatcher (RX)

Cilj:
-----
Provjeriti da RX "inverse rate matching" (DeRateMatcher) radi ispravno
u scenariju gdje TX radi repeat (ponavljanje bitova).

TX logika (iz PBCHEncoder.rate_match):
--------------------------------------
Ako je ulaz dužine N manji od E, TX pravi:
    big = tile(bits, reps=ceil(E/N))
    out = big[:E]

Dakle, svaki originalni bit se pojavi više puta u izlazu.
RX (DeRateMatcher) treba uraditi "akumulaciju" / majority vote:
- grupiše ponovljene instance koje pripadaju istom originalnom bitu
- vrati procjenu originalnih N bitova (0/1)

Test strategija:
----------------
HAPPY (10 testova):
- Generiši originalne bitove dužine N
- TX napravi repeat output dužine E
- Na RX strani namjerno okrenemo mali broj ponovljenih bitova (šum)
  ali tako da većina i dalje ostane tačna
- DeRateMatcher treba vratiti originalne bitove

UNHAPPY (10 testova):
- pogrešni tipovi / dimenzije / prazni ulazi
- ne-binarne vrijednosti
- mismatch očekivanih dužina
"""

from __future__ import annotations

import numpy as np
import pytest

from transmitter.pbch import PBCHEncoder

# Ovdje promijeniti import ako bude drugačiji path 
# Primjer: from receiver.derate_matcher import DeRateMatcher
from receiver.derate_matcher import DeRateMatcher


# ---------------------------------------------------------------------
# Pomoćne funkcije
# ---------------------------------------------------------------------

def tx_repeat_rate_match(bits: np.ndarray, E: int) -> np.ndarray:
    """
    TX repeat logika identična PBCHEncoder.rate_match za slučaj N < E.

    Parametri
    ----------
    bits : np.ndarray
        Ulazni bitovi (0/1) dužine N.
    E : int
        Ciljna dužina nakon rate-matchinga.

    Povratna vrijednost
    -------------------
    np.ndarray
        Niz dužine E nakon ponavljanja (tile + trunc).
    """
    enc = PBCHEncoder(verbose=False)
    return enc.rate_match(bits, E=E)


def _call_deratematcher(rx_bits_E: np.ndarray, N: int, E: int) -> np.ndarray:
    """
    Adapter koji poziva DeRateMatcher bez obzira kako si imenovala metodu.

    Parametri
    ----------
    rx_bits_E : np.ndarray
        RX bitovi dužine E (ponavljani).
    N : int
        Originalna dužina prije rate-matchinga.
    E : int
        Dužina nakon rate-matchinga.

    Povratna vrijednost
    -------------------
    np.ndarray
        Procijenjeni bitovi dužine N.

    Raises
    ------
    AttributeError
        Ako se ne može naći nijedna očekivana metoda.
    """
    dr = DeRateMatcher()

    # Najčešća imena metoda (probamo redom)
    candidates = [
        ("derate_match", (rx_bits_E, N, E)),
        ("inverse_rate_match", (rx_bits_E, N, E)),
        ("decode", (rx_bits_E, N, E)),
        ("apply", (rx_bits_E, N, E)),
        ("run", (rx_bits_E, N, E)),
    ]

    for name, args in candidates:
        if hasattr(dr, name):
            out = getattr(dr, name)(*args)
            return np.asarray(out, dtype=np.uint8).flatten()

    raise AttributeError(
        "Ne mogu naći metodu u DeRateMatcher. Očekujem npr: derate_match / apply / decode / inverse_rate_match."
    )


def _flip_some_repeats(repeated_bits: np.ndarray, flip_indices: np.ndarray) -> np.ndarray:
    """
    Okreće (flip) bitove na zadatim indeksima: 0->1, 1->0.

    Parametri
    ----------
    repeated_bits : np.ndarray
        Bitovi dužine E.
    flip_indices : np.ndarray
        Indeksi gdje se rade flipovi.

    Povratna vrijednost
    -------------------
    np.ndarray
        Novi niz sa flipovanim bitovima.
    """
    y = repeated_bits.copy().astype(np.uint8)
    y[flip_indices] ^= 1
    return y


# ---------------------------------------------------------------------
# HAPPY testovi (10) – repeat + majority/akumulacija
# ---------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_deratematcher_happy_repeat_majority(seed: int) -> None:
    """
    HAPPY:
    TX radi repeat (N < E), a RX DeRateMatcher radi majority/akumulaciju.

    U testu ubacujemo mali broj flipova (grešaka) u ponovljenim bitovima,
    ali tako da većina za svaki originalni bit ostane tačna.
    """
    rng = np.random.default_rng(seed)

    # Biramo N i E tako da TX sigurno ide u repeat granu (N < E).
    # Primjer gdje je E/N cijeli (svaki bit se ponovi jednak broj puta).
    N = 40
    E = 160  # 4 ponavljanja po bitu

    bits_N = rng.integers(0, 2, size=N, dtype=np.uint8)
    tx_bits_E = tx_repeat_rate_match(bits_N, E=E).astype(np.uint8)

    # Flipujemo najviše 1 bit po "grupi" od 4 ponavljanja (majority ostaje tačna).
    flip_idx = []
    for i in range(N):
        group = np.arange(i * 4, i * 4 + 4)
        # flip 0 ili 1 poziciju po grupi
        if rng.random() < 0.7:
            flip_idx.append(int(rng.choice(group)))
    flip_idx = np.array(flip_idx, dtype=int)

    rx_bits_E = _flip_some_repeats(tx_bits_E, flip_idx)

    # RX inverse rate-match (majority)
    rec_bits_N = _call_deratematcher(rx_bits_E, N=N, E=E)

    assert rec_bits_N.shape == (N,)
    assert np.array_equal(rec_bits_N, bits_N)


@pytest.mark.parametrize("seed", range(10))
def test_deratematcher_happy_repeat_non_divisible(seed: int) -> None:
    """
    HAPPY:
    Repeat slučaj gdje E nije djeljivo sa N (zadnje grupe imaju manje ponavljanja).

    Ovo testira da DeRateMatcher pravilno radi akumulaciju i kad
    neke originalne pozicije dobiju različit broj ponavljanja.
    """
    rng = np.random.default_rng(seed)

    # N=5, E=12 -> TX tile 3 puta (15) pa uzme prvih 12:
    # b0,b1 se pojave 3 puta, b2,b3,b4 se pojave 2 puta
    N = 5
    E = 12

    bits_N = rng.integers(0, 2, size=N, dtype=np.uint8)
    tx_bits_E = tx_repeat_rate_match(bits_N, E=E).astype(np.uint8)

    # Flipujemo maksimalno 1 ponavljanje po originalnom bitu,
    # tako da i za grupe od 2 (b2,b3,b4) većina ostane ista.
    # (Za 2 ponavljanja majority je osjetljiv, pa flipujemo najviše 0 u tim grupama.)
    # U ovom konkretnom E=12, indeksi pripadnosti su:
    # [0..4]=b0..b4, [5..9]=b0..b4, [10]=b0, [11]=b1
    safe_flip = []
    # flipujemo samo u b0/b1 grupama (jer imaju 3 ponavljanja)
    # b0 pojavnosti: idx 0,5,10 ; b1: 1,6,11
    if rng.random() < 0.8:
        safe_flip.append(int(rng.choice([0, 5, 10])))
    if rng.random() < 0.8:
        safe_flip.append(int(rng.choice([1, 6, 11])))

    rx_bits_E = _flip_some_repeats(tx_bits_E, np.array(safe_flip, dtype=int))

    rec_bits_N = _call_deratematcher(rx_bits_E, N=N, E=E)

    assert rec_bits_N.shape == (N,)
    assert np.array_equal(rec_bits_N, bits_N)


# ---------------------------------------------------------------------
# UNHAPPY testovi (10) – neispravni ulazi
# ---------------------------------------------------------------------

@pytest.mark.parametrize("case", range(10))
def test_deratematcher_unhappy_invalid_inputs(case: int) -> None:
    """
    UNHAPPY:
    10 različitih neispravnih ulaza za DeRateMatcher.
    """
    dr = DeRateMatcher()

    # Helper za direktan poziv (ako ima "derate_match", koristimo njega, inače adapter)
    def call(rx_bits_E, N, E):
        if hasattr(dr, "derate_match"):
            return np.asarray(dr.derate_match(rx_bits_E, N, E), dtype=np.uint8).flatten()
        return _call_deratematcher(rx_bits_E, N=N, E=E)

    if case == 0:
        # Prazan ulaz
        with pytest.raises(Exception):
            call(np.array([], dtype=np.uint8), N=10, E=20)

    elif case == 1:
        # N = 0 nije validno
        with pytest.raises(Exception):
            call(np.ones(10, dtype=np.uint8), N=0, E=10)

    elif case == 2:
        # E manji od dužine rx_bits_E (mismatch)
        with pytest.raises(Exception):
            call(np.ones(20, dtype=np.uint8), N=10, E=10)

    elif case == 3:
        # rx_bits_E nije 1D
        with pytest.raises(Exception):
            call(np.ones((2, 10), dtype=np.uint8), N=10, E=20)

    elif case == 4:
        # rx_bits_E real float, ali sa vrijednostima koje nisu 0/1
        with pytest.raises(Exception):
            call(np.array([0.1, 0.9, 1.5, -2.0], dtype=np.float32), N=2, E=4)

    elif case == 5:
        # rx_bits_E sadrži NaN
        with pytest.raises(Exception):
            call(np.array([0, 1, np.nan, 1], dtype=np.float32), N=2, E=4)

    elif case == 6:
        # rx_bits_E sadrži Inf
        with pytest.raises(Exception):
            call(np.array([0, 1, np.inf, 1], dtype=np.float32), N=2, E=4)

    elif case == 7:
        # rx_bits_E sadrži vrijednosti mimo {0,1}
        with pytest.raises(Exception):
            call(np.array([0, 1, 2, 1, 0], dtype=np.uint8), N=3, E=5)

    elif case == 8:
        # N veći od E (nema smisla za repeat)
        with pytest.raises(Exception):
            call(np.array([0, 1, 1, 0], dtype=np.uint8), N=10, E=4)

    elif case == 9:
        # E ne odgovara dužini rx_bits_E
        with pytest.raises(Exception):
            call(np.array([0, 1, 1, 0], dtype=np.uint8), N=2, E=10)
