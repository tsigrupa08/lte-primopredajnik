import numpy as np
import pytest

from transmitter.convolutional import ConvolutionalEncoder


def _manual_conv_encode(bits, K, generators_octal, tail_biting=False):
    """
    Referentna (manualna) implementacija za testiranje.
    Radi isto što i encoder, ali pisano jasno radi verifikacije.
    """
    u = np.asarray(bits, dtype=np.uint8).reshape(-1)
    gens = [int(g) for g in generators_octal]
    n_out = len(gens)

    # generator taps (n_out, K)
    taps = np.zeros((n_out, K), dtype=np.uint8)
    for i, g in enumerate(gens):
        taps[i] = [(g >> (K - 1 - k)) & 1 for k in range(K)]

    N = u.size
    if N == 0:
        return np.empty(0, dtype=np.uint8)

    if tail_biting and N >= (K - 1):
        reg = u[-(K - 1):][::-1].copy()
    else:
        reg = np.zeros(K - 1, dtype=np.uint8)

    out = np.zeros((N, n_out), dtype=np.uint8)

    for i in range(N):
        state = np.empty(K, dtype=np.uint8)
        state[0] = u[i]
        state[1:] = reg
        out[i] = (taps @ state) & 1
        reg[1:] = reg[:-1]
        reg[0] = u[i]

    return out.reshape(-1)


# ================================================================
#                       BASIC / HAPPY PATH
# ================================================================

def test_conv_encoder_empty_input():
    enc = ConvolutionalEncoder()
    y = enc.encode(np.array([], dtype=np.uint8))
    assert isinstance(y, np.ndarray)
    assert y.size == 0
    assert y.dtype == np.uint8


def test_conv_encoder_output_length_and_type():
    enc = ConvolutionalEncoder()
    u = np.random.randint(0, 2, 25, dtype=np.uint8)
    y = enc.encode(u)

    assert y.size == u.size * 3
    assert y.dtype == np.uint8
    assert np.all(np.isin(y, [0, 1]))


def test_conv_encoder_deterministic():
    enc = ConvolutionalEncoder()
    u = np.random.randint(0, 2, 50, dtype=np.uint8)

    y1 = enc.encode(u)
    y2 = enc.encode(u)

    assert np.array_equal(y1, y2)


# ================================================================
#                 KNOWN / REFERENCE BEHAVIOR TESTS
# ================================================================

def test_conv_encoder_matches_manual_reference_default_params():
    """
    Najbitniji test: tvoj encoder mora biti identičan referentnom.
    """
    K = 7
    gens = (0o133, 0o171, 0o164)

    enc = ConvolutionalEncoder(constraint_len=K, generators_octal=gens, tail_biting=False)
    u = np.random.randint(0, 2, 40, dtype=np.uint8)

    y = enc.encode(u)
    y_ref = _manual_conv_encode(u, K, gens, tail_biting=False)

    assert np.array_equal(y, y_ref)


def test_conv_encoder_all_zero_input_produces_all_zero_output():
    """
    Ako su ulazni bitovi svi 0 i reg startuje nulama,
    izlaz mora biti sve 0 (za XOR tapove).
    """
    enc = ConvolutionalEncoder(tail_biting=False)
    u = np.zeros(60, dtype=np.uint8)
    y = enc.encode(u)

    assert y.size == 60 * 3
    assert np.all(y == 0)


# ================================================================
#                         TAIL BITING TESTS
# ================================================================

def test_conv_encoder_tail_biting_matches_manual_reference():
    K = 7
    gens = (0o133, 0o171, 0o164)

    enc = ConvolutionalEncoder(constraint_len=K, generators_octal=gens, tail_biting=True)
    u = np.random.randint(0, 2, 40, dtype=np.uint8)

    y = enc.encode(u)
    y_ref = _manual_conv_encode(u, K, gens, tail_biting=True)

    assert np.array_equal(y, y_ref)


def test_conv_encoder_tail_biting_short_input_falls_back_to_zero_reg():
    """
    Ako je N < K-1, tail_biting ne može uzeti zadnjih K-1 bitova,
    pa treba pasti na nulti reg (kao i kod u encoderu).
    """
    K = 7
    gens = (0o133, 0o171, 0o164)

    u = np.random.randint(0, 2, 3, dtype=np.uint8)  # N=3 < K-1=6

    enc_tb = ConvolutionalEncoder(constraint_len=K, generators_octal=gens, tail_biting=True)
    enc_zero = ConvolutionalEncoder(constraint_len=K, generators_octal=gens, tail_biting=False)

    y_tb = enc_tb.encode(u)
    y_zero = enc_zero.encode(u)

    assert np.array_equal(y_tb, y_zero)


# ================================================================
#                     PARAMETER / GENERATOR TEST
# ================================================================

def test_conv_encoder_custom_generators_still_valid_bits():
    """
    Test da encoder radi i sa drugim generatorima (npr. rate 1/2),
    i da vraća samo 0/1.
    """
    enc = ConvolutionalEncoder(constraint_len=7, generators_octal=(0o133, 0o171), tail_biting=False)
    u = np.random.randint(0, 2, 100, dtype=np.uint8)
    y = enc.encode(u)

    assert y.size == u.size * 2
    assert np.all(np.isin(y, [0, 1]))
