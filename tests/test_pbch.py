# tests/test_pbch_encoder.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ------------------------------------------------------------
# PATH FIX: da import radi kad pytest starta iz root-a projekta
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transmitter.pbch import PBCHEncoder


# ============================================================
# Helpers
# ============================================================

def bits01(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def assert_all_01(arr: np.ndarray):
    u = np.unique(arr.astype(np.uint8))
    assert set(u.tolist()).issubset({0, 1})


# ============================================================
# 1) CRC16 tests
# ============================================================

def test_crc16_length_and_binary_output():
    enc = PBCHEncoder(verbose=False)
    info = bits01(24, seed=1)
    crc = enc.crc16(info)
    assert crc.shape == (16,)
    assert crc.dtype == np.uint8
    assert_all_01(crc)


def test_crc16_deterministic_same_input_same_crc():
    enc = PBCHEncoder(verbose=False)
    info = bits01(24, seed=2)
    crc1 = enc.crc16(info)
    crc2 = enc.crc16(info.copy())
    assert np.array_equal(crc1, crc2)


def test_crc16_changes_when_bit_flipped():
    enc = PBCHEncoder(verbose=False)
    info = bits01(24, seed=3)
    crc1 = enc.crc16(info)
    info2 = info.copy()
    info2[0] ^= 1
    crc2 = enc.crc16(info2)
    assert not np.array_equal(crc1, crc2)


# ============================================================
# 2) Sub-block interleaver tests (internal)
# ============================================================

def test_subblock_interleave_120_shape_and_binary():
    enc = PBCHEncoder(verbose=False)
    b120 = bits01(120, seed=4)
    out = enc._subblock_interleave_120(b120)
    assert out.shape == (120,)
    assert out.dtype == np.uint8
    assert_all_01(out)


def test_subblock_interleave_120_is_permutation_of_input_bits_countwise():
    # Interleaver je permutacija (bez gubitka bitova) po streamovima,
    # pa broj jedinica treba ostati isti.
    enc = PBCHEncoder(verbose=False)
    b120 = bits01(120, seed=5)
    out = enc._subblock_interleave_120(b120)
    assert int(np.sum(out)) == int(np.sum(b120))


def test_subblock_interleave_120_asserts_on_wrong_length():
    enc = PBCHEncoder(verbose=False)
    b119 = bits01(119, seed=6)
    with pytest.raises(AssertionError):
        enc._subblock_interleave_120(b119)


# ============================================================
# 3) Rate matching tests
# ============================================================

def test_rate_match_1920_length_and_binary():
    enc = PBCHEncoder(verbose=False)
    b120 = bits01(120, seed=7)
    out = enc.rate_match(b120, E=1920)
    assert out.shape == (1920,)
    assert out.dtype == np.uint8
    assert_all_01(out)


def test_rate_match_1728_length_and_binary():
    enc = PBCHEncoder(verbose=False)
    b120 = bits01(120, seed=8)
    out = enc.rate_match(b120, E=1728)
    assert out.shape == (1728,)
    assert out.dtype == np.uint8
    assert_all_01(out)


def test_rate_match_invalid_E_raises():
    enc = PBCHEncoder(verbose=False)
    b120 = bits01(120, seed=9)
    with pytest.raises(ValueError):
        enc.rate_match(b120, E=100)


def test_rate_match_1920_is_16_repetitions_of_interleaved():
    enc = PBCHEncoder(verbose=False)
    b120 = bits01(120, seed=10)
    inter = enc._subblock_interleave_120(b120)
    out = enc.rate_match(b120, E=1920)
    expected = np.tile(inter, 16).astype(np.uint8)
    assert np.array_equal(out, expected)


def test_rate_match_1728_pattern_is_14_tiles_plus_first48():
    enc = PBCHEncoder(verbose=False)
    b120 = bits01(120, seed=11)
    inter = enc._subblock_interleave_120(b120)
    out = enc.rate_match(b120, E=1728)
    expected = np.concatenate((np.tile(inter, 14), inter[:48])).astype(np.uint8)
    assert np.array_equal(out, expected)


# ============================================================
# 4) Gold sequence (scrambling) tests
# ============================================================

def test_gold_sequence_length_and_binary():
    enc = PBCHEncoder(verbose=False)
    c = enc.gold_sequence_pbch(c_init=0, length=1920)
    assert c.shape == (1920,)
    assert c.dtype == np.uint8
    assert_all_01(c)


def test_gold_sequence_deterministic_same_cinit_same_seq():
    enc = PBCHEncoder(verbose=False)
    c1 = enc.gold_sequence_pbch(c_init=123, length=256)
    c2 = enc.gold_sequence_pbch(c_init=123, length=256)
    assert np.array_equal(c1, c2)


def test_gold_sequence_diff_cinit_diff_seq():
    enc = PBCHEncoder(verbose=False)
    c1 = enc.gold_sequence_pbch(c_init=1, length=256)
    c2 = enc.gold_sequence_pbch(c_init=2, length=256)
    assert not np.array_equal(c1, c2)


def test_gold_sequence_invalid_cinit_raises():
    enc = PBCHEncoder(verbose=False)
    with pytest.raises(AssertionError):
        enc.gold_sequence_pbch(c_init=504, length=10)
    with pytest.raises(AssertionError):
        enc.gold_sequence_pbch(c_init=-1, length=10)


# ============================================================
# 5) QPSK mapping tests
# ============================================================

def test_qpsk_output_dtype_and_length_even_bits():
    enc = PBCHEncoder(verbose=False)
    b = bits01(10, seed=12)  # 10 bits -> 5 symbols
    s = enc.qpsk(b)
    assert s.shape == (5,)
    assert s.dtype == np.complex64


def test_qpsk_appends_zero_if_odd_number_of_bits():
    enc = PBCHEncoder(verbose=False)
    b = bits01(9, seed=13)  # 9 bits -> append 0 -> 10 bits -> 5 symbols
    s = enc.qpsk(b)
    assert s.shape == (5,)
    # sanity: magnitude should be ~1 (normalized)
    mags = np.abs(s).astype(np.float64)
    assert np.allclose(mags, 1.0, atol=1e-6)


def test_qpsk_gray_mapping_known_points():
    enc = PBCHEncoder(verbose=False)
    # (b0,b1) -> (I,Q): 0->+1, 1->-1
    # 00 -> (+1,+1)/sqrt2
    # 01 -> (+1,-1)/sqrt2
    # 10 -> (-1,+1)/sqrt2
    # 11 -> (-1,-1)/sqrt2
    b = np.array([0,0, 0,1, 1,0, 1,1], dtype=np.uint8)
    s = enc.qpsk(b)
    ref = np.array(
        [(1+1j), (1-1j), (-1+1j), (-1-1j)],
        dtype=np.complex64
    ) / np.sqrt(2.0)
    assert np.allclose(s, ref, atol=1e-6)


def test_qpsk_constellation_only_four_points():
    enc = PBCHEncoder(verbose=False)
    b = bits01(100, seed=14)
    s = enc.qpsk(b)
    # normalizuj na najbliže tačke
    pts = np.unique(np.round(s.real, 6) + 1j*np.round(s.imag, 6))
    assert len(pts) <= 4


# ============================================================
# 6) Full encode() chain tests
# ============================================================

def test_encode_outputs_960_symbols_complex64():
    enc = PBCHEncoder(verbose=False, pci=0, enable_scrambling=True)
    info = bits01(24, seed=15)
    syms = enc.encode(info)
    assert syms.shape == (960,)
    assert syms.dtype == np.complex64


def test_encode_deterministic_when_same_input_and_same_pci_and_scrambling_on():
    enc = PBCHEncoder(verbose=False, pci=7, enable_scrambling=True)
    info = bits01(24, seed=16)
    s1 = enc.encode(info)
    s2 = enc.encode(info.copy())
    assert np.allclose(s1, s2, atol=0.0)  # identično bi trebalo biti


def test_encode_changes_with_pci_when_scrambling_on():
    info = bits01(24, seed=17)
    e1 = PBCHEncoder(verbose=False, pci=0, enable_scrambling=True).encode(info)
    e2 = PBCHEncoder(verbose=False, pci=1, enable_scrambling=True).encode(info)
    assert not np.allclose(e1, e2)


def test_encode_same_with_different_pci_when_scrambling_off():
    info = bits01(24, seed=18)
    e1 = PBCHEncoder(verbose=False, pci=0, enable_scrambling=False).encode(info)
    e2 = PBCHEncoder(verbose=False, pci=503, enable_scrambling=False).encode(info)
    assert np.allclose(e1, e2)


def test_encode_asserts_on_wrong_info_length():
    enc = PBCHEncoder(verbose=False)
    info = bits01(23, seed=19)
    with pytest.raises(AssertionError):
        enc.encode(info)


def test_encode_accepts_python_list_input_length_24():
    enc = PBCHEncoder(verbose=False)
    info = bits01(24, seed=20).tolist()
    syms = enc.encode(info)
    assert syms.shape == (960,)


def test_encode_output_has_unit_magnitude_constellation():
    enc = PBCHEncoder(verbose=False)
    info = bits01(24, seed=21)
    syms = enc.encode(info)
    mags = np.abs(syms).astype(np.float64)
    # zbog normalizacije u qpsk, svi simboli magnitude ~1
    assert np.allclose(mags, 1.0, atol=1e-5)


def test_encode_verbose_does_not_change_output():
    info = bits01(24, seed=22)
    s1 = PBCHEncoder(verbose=False, pci=3, enable_scrambling=True).encode(info)
    s2 = PBCHEncoder(verbose=True, pci=3, enable_scrambling=True).encode(info)
    assert np.allclose(s1, s2, atol=0.0)
