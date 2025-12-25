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

Implementacija DeRateMatcher u ovom projektu:
--------------------------------------------
- DeRateMatcher(E_rx, N_coded)
- accumulate(bits_rx, soft=True/False)
  * radi average po indeksima modulo N_coded
  * soft=False radi hard decision preko (soft_bits >= 0.5)

Test strategija:
----------------
HAPPY (10 testova):
- Generiši originalne bitove dužine N
- TX napravi repeat output dužine E (tile + trunc)
- Na RX strani namjerno okrenemo mali broj ponovljenih bitova (šum)
  ali tako da većina i dalje ostane tačna
- DeRateMatcher treba vratiti originalne bitove

HAPPY (10 testova):
- Repeat slučaj gdje E nije djeljivo sa N (neke pozicije imaju 3 puta, neke 2 puta)
- Flipujemo samo u grupama koje imaju >=3 ponavljanja da majority ostane tačna

UNHAPPY (10 testova):
- "problematični" ulazi
Napomena: pošto trenutna implementacija nema eksplicitnu validaciju ulaza,
ne očekujemo uvijek exception; umjesto toga provjeravamo posljedice.
"""

from __future__ import annotations

import numpy as np
import pytest

from transmitter.pbch import PBCHEncoder
from receiver.de_rate_matching import DeRateMatcher


# ---------------------------------------------------------------------
# Pomoćne funkcije
# ---------------------------------------------------------------------

def tx_repeat_rate_match(bits: np.ndarray, E: int) -> np.ndarray:
    """
    TX repeat logika identična PBCHEncoder.rate_match za slučaj N < E.

    Parameters
    ----------
    bits : np.ndarray
        Ulazni bitovi (0/1) dužine N.
    E : int
        Ciljna dužina nakon rate-matchinga.

    Returns
    -------
    np.ndarray
        Niz dužine E nakon ponavljanja (tile + trunc).
    """
    enc = PBCHEncoder(verbose=False)
    return enc.rate_match(bits, E=E)


def rx_derate_majority(rx_bits_E: np.ndarray, N: int) -> np.ndarray:
    """
    RX de-rate-match koristeći trenutnu implementaciju:
    DeRateMatcher(E_rx, N_coded=N).accumulate(rx_bits_E, soft=False)

    Parameters
    ----------
    rx_bits_E : np.ndarray
        Primljeni bitovi dužine E (ponavljani).
    N : int
        Originalna dužina prije rate-matchinga.

    Returns
    -------
    np.ndarray
        Hard decision bitovi dužine N (0/1).
    """
    dr = DeRateMatcher(E_rx=1.0, N_coded=N)
    out = dr.accumulate(rx_bits_E, soft=False)
    return np.asarray(out, dtype=np.uint8).flatten()


def _flip_some_repeats(repeated_bits: np.ndarray, flip_indices: np.ndarray) -> np.ndarray:
    """
    Okreće (flip) bitove na zadatim indeksima: 0->1, 1->0.
    """
    y = repeated_bits.copy().astype(np.uint8)
    if flip_indices.size > 0:
        y[flip_indices] ^= 1
    return y


def _repeat_positions_for_bit(i: int, N: int, E: int) -> np.ndarray:
    """
    Vrati sve pozicije u TX tile izlazu (dužine E) gdje se pojavi originalni bit i.

    Za tile + trunc obrazac:
        out[k] = bits[k % N]
    => bit i se pojavljuje na indeksima: i, i+N, i+2N, ... < E

    Parameters
    ----------
    i : int
        Indeks originalnog bita (0..N-1)
    N : int
        Dužina originalnog niza
    E : int
        Dužina nakon rate-matchinga

    Returns
    -------
    np.ndarray
        Indeksi pojavljivanja bita i u TX izlazu.
    """
    return np.arange(i, E, N, dtype=int)


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

    N = 40
    E = 160  # prosječno 4 ponavljanja po bitu

    bits_N = rng.integers(0, 2, size=N, dtype=np.uint8)
    tx_bits_E = tx_repeat_rate_match(bits_N, E=E).astype(np.uint8)

    # Flipujemo najviše 1 ponavljanje po originalnom bitu
    flip_idx: list[int] = []
    for i in range(N):
        group = _repeat_positions_for_bit(i, N=N, E=E)
        # sigurno: group ima 4 elementa uN=40,E=160, ali nek bude robustno
        if group.size > 0 and rng.random() < 0.7:
            flip_idx.append(int(rng.choice(group)))

    rx_bits_E = _flip_some_repeats(tx_bits_E, np.asarray(flip_idx, dtype=int))

    rec_bits_N = rx_derate_majority(rx_bits_E, N=N)

    assert rec_bits_N.shape == (N,)
    assert np.array_equal(rec_bits_N, bits_N)


@pytest.mark.parametrize("seed", range(10))
def test_deratematcher_happy_repeat_non_divisible(seed: int) -> None:
    """
    HAPPY:
    Repeat slučaj gdje E nije djeljivo sa N.

    Primjer: N=5, E=12:
      pojavljivanja:
        i=0 -> idx [0,5,10] (3 puta)
        i=1 -> idx [1,6,11] (3 puta)
        i=2 -> idx [2,7]    (2 puta)
        i=3 -> idx [3,8]    (2 puta)
        i=4 -> idx [4,9]    (2 puta)

    Flipujemo samo u grupama koje imaju >=3 ponavljanja (da majority ostane tačna).
    """
    rng = np.random.default_rng(seed)

    N = 5
    E = 12

    bits_N = rng.integers(0, 2, size=N, dtype=np.uint8)
    tx_bits_E = tx_repeat_rate_match(bits_N, E=E).astype(np.uint8)

    flip_idx: list[int] = []

    # Flipujemo najviše 1 u grupama sa >=3 ponavljanja
    for i in range(N):
        group = _repeat_positions_for_bit(i, N=N, E=E)
        if group.size >= 3 and rng.random() < 0.8:
            flip_idx.append(int(rng.choice(group)))

    rx_bits_E = _flip_some_repeats(tx_bits_E, np.asarray(flip_idx, dtype=int))

    rec_bits_N = rx_derate_majority(rx_bits_E, N=N)

    assert rec_bits_N.shape == (N,)
    assert np.array_equal(rec_bits_N, bits_N)


# ---------------------------------------------------------------------
# UNHAPPY testovi (10) – neispravni / problematični ulazi
# ---------------------------------------------------------------------

@pytest.mark.parametrize("case", range(10))
def test_deratematcher_unhappy_invalid_inputs(case: int) -> None:
    """
    UNHAPPY:
    10 različitih neispravnih ulaza.

    Napomena: Implementacija DeRateMatcher trenutno nema eksplicitnu validaciju,
    pa ne očekujemo uvijek exception. Umjesto toga, provjeravamo posljedice.
    """
    def call_soft(rx_bits_E, N):
        dr = DeRateMatcher(E_rx=1.0, N_coded=N)
        return np.asarray(dr.accumulate(rx_bits_E, soft=True), dtype=float).flatten()

    def call_hard(rx_bits_E, N):
        dr = DeRateMatcher(E_rx=1.0, N_coded=N)
        return np.asarray(dr.accumulate(rx_bits_E, soft=False), dtype=np.uint8).flatten()

    if case == 0:
        # Prazan ulaz: implementacija vrati nule dužine N (nema exception)
        out = call_soft(np.array([], dtype=np.float32), N=10)
        assert out.shape == (10,)
        assert np.all(out == 0.0)

    elif case == 1:
        # N=0: trenutno može dati warning/Inf/NaN ili exception (zavisno od numpy ponašanja).
        # Test prihvata oba: ili exception, ili "nevalidan" rezultat.
        try:
            out = call_soft(np.ones(10, dtype=np.float32), N=0)
            # Ako nije bacilo exception, očekujemo da rezultat bude prazan (minlength=0) ili da sadrži NaN/Inf
            assert out.size == 0 or (np.isnan(out).any() or np.isinf(out).any())
        except Exception:
            assert True

    elif case == 2:
        # ulaz može biti bilo koje dužine; provjera: radi i vrati dužinu N
        out = call_hard(np.ones(20, dtype=np.uint8), N=10)
        assert out.shape == (10,)
        assert set(np.unique(out)).issubset({0, 1})

    elif case == 3:
        # rx_bits_E nije 1D -> bincount weights mora biti 1D, očekujemo exception
        with pytest.raises(Exception):
            _ = call_soft(np.ones((2, 10), dtype=np.float32), N=10)

    elif case == 4:
        # float ne-binarno -> hard decision i dalje daje 0/1
        out = call_hard(np.array([0.1, 0.9, 1.5, -2.0], dtype=np.float32), N=2)
        assert out.shape == (2,)
        assert set(np.unique(out)).issubset({0, 1})

    elif case == 5:
        # NaN -> u soft režimu očekujemo NaN u izlazu (propagacija)
        out = call_soft(np.array([0.0, 1.0, np.nan, 1.0], dtype=np.float32), N=2)
        assert out.shape == (2,)
        assert np.isnan(out).any()

    elif case == 6:
        # Inf -> u soft režimu očekujemo Inf u izlazu
        out = call_soft(np.array([0.0, 1.0, np.inf, 1.0], dtype=np.float32), N=2)
        assert out.shape == (2,)
        assert np.isinf(out).any()

    elif case == 7:
        # vrijednosti mimo {0,1} -> soft rezultat može biti >1 (bez validacije)
        out = call_soft(np.array([0, 1, 2, 1, 0], dtype=np.float32), N=3)
        assert out.shape == (3,)
        assert np.max(out) > 1.0

    elif case == 8:
        # N > E: dio pozicija nema nijedno pojavljivanje -> ostaju 0 (po implementaciji)
        out = call_hard(np.array([0, 1, 1, 0], dtype=np.uint8), N=10)
        assert out.shape == (10,)
        assert np.all(out[4:] == 0)

    elif case == 9:
        # bilo koja dužina radi i daje dužinu N
        out = call_hard(np.array([0, 1, 1, 0], dtype=np.uint8), N=2)
        assert out.shape == (2,)
        assert set(np.unique(out)).issubset({0, 1})
