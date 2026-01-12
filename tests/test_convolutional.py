import os
import sys
import numpy as np
import pytest

# ------------------------------------------------------------
# Ensure project root is on sys.path (radi kad pytest pokreće iz root-a ili drugih foldera)
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transmitter.convolutional import ConvolutionalEncoder


# =========================
# Hardcoded expected vectors
# =========================

# Default taps for K=7, generators: (0o133, 0o171, 0o165)
EXPECTED_TAPS_DEFAULT = np.array(
    [
        [1, 0, 1, 1, 0, 1, 1],  # 133(octal)
        [1, 1, 1, 1, 0, 0, 1],  # 171(octal)
        [1, 1, 1, 0, 1, 0, 1],  # 165(octal)
    ],
    dtype=np.uint8,
)

# Known encode outputs (tail_biting=False)
U1 = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
Y1_EXPECT = np.array(
    [1, 1, 1,  0, 1, 1,  0, 0, 0,  0, 1, 0,  1, 0, 1,  1, 0, 1,  1, 1, 1],
    dtype=np.uint8
)

# Known encode outputs for another vector
U2 = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
Y2_EXPECT = np.array(
    [1, 1, 1,  0, 1, 1,  0, 0, 0,  0, 1, 0,  1, 0, 1,  0, 1, 0,  0, 1, 1],
    dtype=np.uint8
)

# Tail-biting expected output for U2 (tail_biting=True)
Y2_TB_EXPECT = np.array(
    [1, 0, 1,  1, 1, 0,  1, 1, 0,  1, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 1],
    dtype=np.uint8
)


# =========================
# Tests
# =========================

def test_default_params_sanity():
    enc = ConvolutionalEncoder()
    assert enc.K == 7
    assert tuple(enc.generators.tolist()) == (0o133, 0o171, 0o165)
    assert enc.n_out == 3
    assert enc.tail_biting is False


def test_taps_matrix_default_polynomials():
    enc = ConvolutionalEncoder()
    np.testing.assert_array_equal(enc.taps.astype(np.uint8), EXPECTED_TAPS_DEFAULT)


def test_encode_output_length_rate_one_third():
    enc = ConvolutionalEncoder()
    u = np.random.randint(0, 2, size=50, dtype=np.uint8)
    y = enc.encode(u)
    assert y.size == u.size * 3


def test_encode_output_is_1d():
    enc = ConvolutionalEncoder()
    u = np.random.randint(0, 2, size=10, dtype=np.uint8)
    y = enc.encode(u)
    assert y.ndim == 1


def test_encode_output_bits_are_binary():
    enc = ConvolutionalEncoder()
    u = np.random.randint(0, 2, size=100, dtype=np.uint8)
    y = enc.encode(u)
    assert set(np.unique(y)).issubset({0, 1})


def test_known_vector_no_tail_biting_u1():
    enc = ConvolutionalEncoder(tail_biting=False)
    y = enc.encode(U1)
    np.testing.assert_array_equal(y, Y1_EXPECT)


def test_known_vector_no_tail_biting_u2():
    enc = ConvolutionalEncoder(tail_biting=False)
    y = enc.encode(U2)
    np.testing.assert_array_equal(y, Y2_EXPECT)


def test_known_vector_tail_biting_u2():
    enc = ConvolutionalEncoder(tail_biting=True)
    y = enc.encode(U2)
    np.testing.assert_array_equal(y, Y2_TB_EXPECT)


def test_tail_biting_changes_output_for_nontrivial_sequence():
    u = U2.copy()
    y_no = ConvolutionalEncoder(tail_biting=False).encode(u)
    y_tb = ConvolutionalEncoder(tail_biting=True).encode(u)
    assert not np.array_equal(y_no, y_tb)


def test_empty_input_returns_empty():
    enc = ConvolutionalEncoder()
    y = enc.encode([])
    assert isinstance(y, np.ndarray)
    assert y.size == 0


def test_accepts_list_tuple_numpy_inputs_same_result():
    enc = ConvolutionalEncoder()
    u_list = [1, 0, 1, 1, 0, 1, 0]
    u_tuple = tuple(u_list)
    u_np = np.array(u_list, dtype=np.uint8)

    y1 = enc.encode(u_list)
    y2 = enc.encode(u_tuple)
    y3 = enc.encode(u_np)

    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(y1, y3)


def test_nonbinary_inputs_use_lsb_effectively():
    # 2 -> 0 (LSB=0), 3 -> 1 (LSB=1)
    enc = ConvolutionalEncoder()
    u_weird = np.array([2, 3, 2, 3, 3, 2], dtype=np.uint8)
    u_bits  = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)

    y_weird = enc.encode(u_weird)
    y_bits = enc.encode(u_bits)
    np.testing.assert_array_equal(y_weird, y_bits)


def test_custom_k_and_generators_known_output():
    # Mali sanity za generalnost: K=3, rate=1/2, gens=(7,5) oktalno
    enc = ConvolutionalEncoder(constraint_len=3, generators_octal=(0o7, 0o5), tail_biting=False)
    u = np.array([1, 0, 1, 1], dtype=np.uint8)
    y = enc.encode(u)
    y_expected = np.array([1, 1,  1, 0,  0, 0,  0, 1], dtype=np.uint8)
    np.testing.assert_array_equal(y, y_expected)


def test_deterministic_same_input_same_output():
    enc = ConvolutionalEncoder()
    u = np.random.randint(0, 2, size=80, dtype=np.uint8)
    y1 = enc.encode(u)
    y2 = enc.encode(u)
    np.testing.assert_array_equal(y1, y2)


def test_first_symbol_matches_zero_state_expectation():
    # Za zero init reg, prvi izlaz za u0=1 treba biti taps[:,0] (svi su 1) => [1,1,1]
    enc = ConvolutionalEncoder(tail_biting=False)
    u = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
    y = enc.encode(u)
    np.testing.assert_array_equal(y[:3], np.array([1, 1, 1], dtype=np.uint8))


def test_linearity_over_gf2_no_tail_biting():
    # Konvolucioni kod je linearan (za zero init stanje): enc(a^b) == enc(a)^enc(b)
    enc = ConvolutionalEncoder(tail_biting=False)
    a = np.random.randint(0, 2, size=50, dtype=np.uint8)
    b = np.random.randint(0, 2, size=50, dtype=np.uint8)
    lhs = enc.encode(a ^ b)
    rhs = enc.encode(a) ^ enc.encode(b)
    np.testing.assert_array_equal(lhs, rhs)
