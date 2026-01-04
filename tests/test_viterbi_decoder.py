"""
Test suite za ViterbiDecoder (hard-decision).

Ciljevi testiranja:
- provjera ispravnosti (happy path)
- robusnost na neidealne ulaze (unhappy path)
- ponašanje na rubnim slučajevima
- determinističnost i konzistentnost izlaza

Testovi: 
- ispituju isključivo ponašanje sistema
"""

import numpy as np
import pytest

from transmitter.convolutional import ConvolutionalEncoder
from receiver.viterbi_decoder import ViterbiDecoder


# ============================================================
# POMOĆNE FUNKCIJE
# ============================================================

def make_encoder_decoder(rate=1/3):
    """
    Kreira standardni konvolucijski encoder i Viterbi dekoder.

    Parametri
    ----------
    rate : float
        Kodna stopa (informativna za dekoder).

    Povrat
    ------
    enc : ConvolutionalEncoder
    dec : ViterbiDecoder
    """
    enc = ConvolutionalEncoder(
        constraint_len=7,
        generators_octal=(0o133, 0o171, 0o164),
        tail_biting=False
    )

    dec = ViterbiDecoder(
        constraint_len=7,
        generators=[0o133, 0o171, 0o164],
        rate=rate
    )

    return enc, dec


# ============================================================
# HAPPY PATH TESTOVI
# ============================================================

@pytest.mark.parametrize("length", [1, 2, 5, 10, 20, 50, 100])
def test_viterbi_happy_path_exact_recovery(length):
    """
    HAPPY PATH:
    Bez šuma i sa ispravnim generatorima,
    dekoder mora TAČNO vratiti originalne bitove.
    """
    u = np.random.randint(0, 2, length, dtype=np.uint8)

    enc, dec = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)
    u_hat = dec.decode(coded)

    assert np.array_equal(u, u_hat)


def test_viterbi_single_bit_input():
    """
    HAPPY PATH (minimalni slučaj):
    Dekodiranje jednog bita.
    """
    u = np.array([1], dtype=np.uint8)

    enc, dec = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)
    u_hat = dec.decode(coded)

    assert u_hat.size == 1
    assert u_hat[0] in (0, 1)


# ============================================================
# EDGE CASE TESTOVI
# ============================================================

def test_viterbi_empty_input():
    """
    EDGE CASE:
    Prazan ulaz → prazan izlaz.
    """
    _, dec = make_encoder_decoder(rate=1/3)

    u_hat = dec.decode(np.array([], dtype=np.uint8))

    assert isinstance(u_hat, np.ndarray)
    assert u_hat.size == 0


def test_viterbi_input_shorter_than_one_symbol():
    """
    EDGE CASE:
    Manje bitova nego n_out → nema dekodiranja.
    """
    _, dec = make_encoder_decoder(rate=1/3)

    # n_out = 3 → samo 2 bita
    u_hat = dec.decode(np.array([1, 0], dtype=np.uint8))

    assert isinstance(u_hat, np.ndarray)
    assert u_hat.size == 0


# ============================================================
# ROBUSTNOST NA DUŽINU
# ============================================================

def test_viterbi_truncates_extra_bits():
    """
    UNHAPPY PATH:
    Višak bitova na ulazu se ignoriše, ali ako
    formira novi puni simbol, dekoder ga dekodira.
    """
    u = np.random.randint(0, 2, 15, dtype=np.uint8)

    enc, dec = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)

    # Dodaj višak bitova
    bad_input = np.concatenate([coded, np.array([1, 0, 1, 1], dtype=np.uint8)])

    u_hat = dec.decode(bad_input)

    expected_len = bad_input.size // dec.n_out

    assert isinstance(u_hat, np.ndarray)
    assert u_hat.size == expected_len


# ============================================================
# ROBUSTNOST NA RATE
# ============================================================

@pytest.mark.parametrize("wrong_rate", [1/2, 0.4, 0.9])
def test_viterbi_ignores_wrong_rate(wrong_rate):
    """
    UNHAPPY PATH:
    Pogrešno specificiran rate NE SMIJE slomiti dekoder.
    """
    u = np.random.randint(0, 2, 20, dtype=np.uint8)

    enc, _ = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)

    dec = ViterbiDecoder(
        constraint_len=7,
        generators=[0o133, 0o171, 0o164],
        rate=wrong_rate
    )

    u_hat = dec.decode(coded)

    assert isinstance(u_hat, np.ndarray)
    assert u_hat.size == u.size


def test_viterbi_rate_does_not_affect_result():
    """
    REGRESIONI TEST:
    Različit rate ne smije promijeniti rezultat
    ako su generatori isti.
    """
    u = np.random.randint(0, 2, 25, dtype=np.uint8)

    enc, _ = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)

    dec1 = ViterbiDecoder(
        constraint_len=7,
        generators=[0o133, 0o171, 0o164],
        rate=1/3
    )
    dec2 = ViterbiDecoder(
        constraint_len=7,
        generators=[0o133, 0o171, 0o164],
        rate=1/2
    )

    u_hat1 = dec1.decode(coded)
    u_hat2 = dec2.decode(coded)

    assert np.array_equal(u_hat1, u_hat2)


# ============================================================
# ROBUSTNOST NA ŠUM
# ============================================================

def test_viterbi_with_small_noise():
    """
    UNHAPPY PATH:
    Mali broj grešaka (bit flip) → dekoder i dalje radi.
    """
    u = np.random.randint(0, 2, 40, dtype=np.uint8)

    enc, dec = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)

    # Flip 5% bitova
    noisy = coded.copy()
    idx = np.random.choice(len(noisy), size=len(noisy) // 20, replace=False)
    noisy[idx] ^= 1

    u_hat = dec.decode(noisy)

    assert isinstance(u_hat, np.ndarray)
    assert u_hat.size == u.size


def test_viterbi_with_heavy_noise():
    """
    EXTREME UNHAPPY PATH:
    Jak šum → dekoder NE SMIJE pucati.
    """
    u = np.random.randint(0, 2, 40, dtype=np.uint8)

    enc, dec = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)

    # Flip ~30% bitova
    noisy = coded.copy()
    idx = np.random.choice(len(noisy), size=len(noisy) // 3, replace=False)
    noisy[idx] ^= 1

    u_hat = dec.decode(noisy)

    assert isinstance(u_hat, np.ndarray)
    assert u_hat.size == u.size


# ============================================================
# KONSISTENTNOST I VALIDNOST IZLAZA
# ============================================================

def test_viterbi_output_is_binary():
    """
    Izlaz dekodera mora sadržavati isključivo 0 i 1.
    """
    u = np.random.randint(0, 2, 30, dtype=np.uint8)

    enc, dec = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)
    u_hat = dec.decode(coded)

    assert set(np.unique(u_hat)).issubset({0, 1})


def test_viterbi_is_deterministic():
    """
    DETERMINISTIČNOST:
    Isti ulaz → isti izlaz.
    """
    u = np.random.randint(0, 2, 25, dtype=np.uint8)

    enc, dec = make_encoder_decoder(rate=1/3)
    coded = enc.encode(u)

    u_hat1 = dec.decode(coded)
    u_hat2 = dec.decode(coded)

    assert np.array_equal(u_hat1, u_hat2)
