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
- TX napravi repeat output dužine E
- Na RX strani namjerno okrenemo mali broj ponovljenih bitova (šum)
  ali tako da većina i dalje ostane tačna
- DeRateMatcher treba vratiti originalne bitove

HAPPY (10 testova):
- Repeat slučaj gdje E nije djeljivo sa N (neke pozicije imaju 3 puta, neke 2 puta)
- Flipujemo samo u grupama koje imaju 3 ponavljanja da majority ostane tačna

UNHAPPY (10 testova):
- neispravni / "problematični" ulazi
Napomena: pošto trenutna implementacija nema eksplicitnu validaciju ulaza,
ne očekujemo uvijek exception; umjesto toga provjeravamo da rezultat pokazuje
"nepoželjne" efekte (npr. propagacija NaN/Inf, ignorisanje mismatch-a E, itd.).
"""

from __future__ import annotations

import numpy as np
import pytest

from transmitter.pbch import PBCHEncoder
from receiver.derate_matcher import DeRateMatcher


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
    out = np.asarray(out, dtype=np.uint8).flatten()
    return out


def _flip_some_repeats(repeated_bits: np.ndarray, flip_indices: np.ndarray) -> np.ndarray:
    """
    Okreće (flip) bitove na zadatim indeksima: 0->1, 1->0.
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

    N = 40
    E = 160  # 4 ponavljanja po bitu

    bits_N = rng.integers(0, 2, size=N, dtype=np.uint8)
    tx_bits_E = tx_repeat_rate_match(bits_N, E=E).astype(np.uint8)

    # Flipujemo najviše 1 bit po "grupi" od 4 ponavljanja (majority ostaje tačna).
    flip_idx = []
    for i in range(N):
        group = np.arange(i * 4, i * 4 + 4)
        if rng.random() < 0.7:
            flip_idx.append(int(rng.choice(group)))
    flip_idx = np.array(flip_idx, dtype=int)

    rx_bits_E = _flip_some_repeats(tx_bits_E, flip_idx)

    rec_bits_N = rx_derate_majority(rx_bits_E, N=N)

    assert rec_bits_N.shape == (N,)
    assert np.array_equal(rec_bits_N, bits_N)


@pytest.mark.parametrize("seed", range(10))
def test_deratematcher_happy_repeat_non_divisible(seed: int) -> None:
    """
    HAPPY:
    Repeat slučaj gdje E nije djeljivo sa N (zadnje grupe imaju manje ponavljanja).

    E=12, N=5 -> b0,b1 se pojave 3 puta, b2,b3,b4 se pojave 2 puta.
    Flipujemo samo u b0/b1 (3 ponavljanja), da majority ostane tačna.
    """
    rng = np.random.default_rng(seed)

    N = 5
    E = 12

    bits_N = rng.integers(0, 2, size=N, dtype=np.uint8)
    tx_bits_E = tx_repeat_rate_match(bits_N, E=E).astype(np.uint8)

    safe_flip = []
    # b0: idx 0,5,10 ; b1: idx 1,6,11
    if rng.random() < 0.8:
        safe_flip.append(int(rng.choice([0, 5, 10])))
    if rng.random() < 0.8:
        safe_flip.append(int(rng.choice([1, 6, 11])))

    rx_bits_E = _flip_some_repeats(tx_bits_E, np.array(safe_flip, dtype=int))

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
    pa ne očekujemo uvijek exception. Umjesto toga, provjeravamo posljedice:
    - NaN/Inf se propagiraju u soft režimu
    - 2D ulaz baca grešku (bincount weights mora biti 1D)
    - N=0 izaziva grešku (modulo 0)
    - N > E (nema repeat-a) vrati niz dužine N gdje nepopunjene pozicije ostanu 0
    - mismatch "E" se ovdje ne testira jer DeRateMatcher ne prima E kao argument
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
        # N=0 nije validno (modulo 0 / bincount minlength 0)
        with pytest.raises(Exception):
            _ = call_soft(np.ones(10, dtype=np.float32), N=0)

    elif case == 2:
        # "Mismatch očekivanih dužina" se ovdje mapira na: ulaz može biti bilo koje dužine.
        # Provjera: radi i vrati dužinu N.
        out = call_hard(np.ones(20, dtype=np.uint8), N=10)
        assert out.shape == (10,)
        assert set(np.unique(out)).issubset({0, 1})

    elif case == 3:
        # rx_bits_E nije 1D -> bincount weights mora biti 1D, očekujemo exception
        with pytest.raises(Exception):
            _ = call_soft(np.ones((2, 10), dtype=np.float32), N=10)

    elif case == 4:
        # real float, ali ne-binarne vrijednosti -> nema exception; hard decision i dalje daje 0/1
        out = call_hard(np.array([0.1, 0.9, 1.5, -2.0], dtype=np.float32), N=2)
        assert out.shape == (2,)
        assert set(np.unique(out)).issubset({0, 1})

    elif case == 5:
        # sadrži NaN -> u soft režimu očekujemo NaN u izlazu
        out = call_soft(np.array([0.0, 1.0, np.nan, 1.0], dtype=np.float32), N=2)
        assert out.shape == (2,)
        assert np.isnan(out).any()

    elif case == 6:
        # sadrži Inf -> u soft režimu očekujemo Inf u izlazu
        out = call_soft(np.array([0.0, 1.0, np.inf, 1.0], dtype=np.float32), N=2)
        assert out.shape == (2,)
        assert np.isinf(out).any()

    elif case == 7:
        # vrijednosti mimo {0,1} -> soft rezultat može biti > 1 (nepoželjno, ali očekivano bez validacije)
        out = call_soft(np.array([0, 1, 2, 1, 0], dtype=np.float32), N=3)
        assert out.shape == (3,)
        assert np.max(out) > 1.0

    elif case == 8:
        # N > E: nema repeat-a, dio pozicija nema nijedno pojavljivanje -> ostaju 0 (po implementaciji)
        out = call_hard(np.array([0, 1, 1, 0], dtype=np.uint8), N=10)
        assert out.shape == (10,)
        # prva 4 mjesta imaju nešto, ostatak ostaje 0
        assert np.all(out[4:] == 0)

    elif case == 9:
        # "E ne odgovara dužini rx_bits_E" nije primjenjivo jer E se ne prosljeđuje.
        # Provjera: bilo koja dužina radi i daje dužinu N.
        out = call_hard(np.array([0, 1, 1, 0], dtype=np.uint8), N=2)
        assert out.shape == (2,)
        assert set(np.unique(out)).issubset({0, 1})
