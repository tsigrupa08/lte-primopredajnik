# tests/test_pbch.py
import numpy as np
import pytest
from transmitter.pbch import PBCHEncoder

# ----------------- Happy Path / Dummy Tests -----------------

def test_happy_path_basic_encode():
    """
    Happy path test: osnovno enkodiranje malog niza info bitova.
    """
    info_bits = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    encoder = PBCHEncoder(target_bits=32, verbose=False)
    symbols = encoder.encode(info_bits)
    
    assert isinstance(symbols, np.ndarray)
    assert symbols.size == 16  # 32 bits / 2 -> 16 QPSK simbola
    assert np.iscomplexobj(symbols)

def test_happy_path_crc_only():
    """
    Happy path test: samo CRC generacija
    """
    info_bits = np.array([1, 0, 1, 0])
    encoder = PBCHEncoder(verbose=False)
    crc = encoder.generate_crc24a(info_bits)
    
    assert isinstance(crc, np.ndarray)
    assert crc.size == 24
    assert np.all(np.isin(crc, [0,1]))

def test_dummy_rate_matching_repeat():
    """
    Dummy test: rate matching ponavlja kraći niz da postigne target_bits
    """
    bits = np.array([0,1,1])
    encoder = PBCHEncoder(target_bits=9, verbose=False)
    rm = encoder.rate_match(bits)
    
    expected = np.tile(bits, 3)
    assert np.array_equal(rm, expected)

def test_dummy_qpsk_map_even_odd():
    """
    Dummy test: QPSK mapping provjera da se pravilno dodaje nula ako je neparno
    """
    bits_odd = np.array([1,0,1])
    encoder = PBCHEncoder(verbose=False)
    symbols = encoder.qpsk_map(bits_odd)
    
    assert symbols.size == 2
    assert np.iscomplexobj(symbols)

# ----------------- CRC Tests -----------------

def test_crc_output_length_and_type():
    info_bits = np.random.randint(0,2,10)
    encoder = PBCHEncoder(verbose=False)
    crc = encoder.generate_crc24a(info_bits)
    
    assert crc.size == 24
    assert np.all(np.isin(crc, [0,1]))

# ----------------- Convolutional Encoder Tests -----------------

def test_conv_encoder_output_length():
    bits = np.array([1,0,1,1])
    encoder = PBCHEncoder(verbose=False)
    coded = encoder._conv_encode_fallback(bits)
    
    # length = 2*N + 12 tail bits
    assert len(coded) == 2*len(bits) + 12
    assert np.all(np.isin(coded, [0,1]))

# ----------------- Rate Matching Tests -----------------

def test_rate_matching_puncture():
    bits = np.arange(10) % 2
    encoder = PBCHEncoder(target_bits=5, verbose=False)
    rm = encoder.rate_match(bits)
    assert rm.size == 5
    assert np.all(np.isin(rm, [0,1]))

def test_rate_matching_repeat_longer():
    bits = np.array([0,1])
    encoder = PBCHEncoder(target_bits=6, verbose=False)
    rm = encoder.rate_match(bits)
    expected = np.tile(bits, 3)
    assert np.array_equal(rm, expected)

# ----------------- QPSK Mapping Tests -----------------

def test_qpsk_mapping_values():
    bits = np.array([0,0,0,1,1,0,1,1])
    encoder = PBCHEncoder(verbose=False)
    symbols = encoder.qpsk_map(bits)
    
    assert symbols.size == 4
    assert np.iscomplexobj(symbols)
    # Energy check: average power ~1
    power = np.mean(np.abs(symbols)**2)
    assert np.isclose(power, 1.0, atol=1e-8)

# ----------------- Full Chain Tests -----------------

def test_full_chain_output_type_and_length():
    info_bits = np.random.randint(0,2,24)
    encoder = PBCHEncoder(target_bits=384, verbose=False)
    symbols = encoder.encode(info_bits)
    
    assert isinstance(symbols, np.ndarray)
    assert symbols.size == 384//2
    assert np.iscomplexobj(symbols)

def test_full_chain_happy_path_values():
    info_bits = np.array([1,0,1,0,1,1])
    encoder = PBCHEncoder(target_bits=12, verbose=False)
    symbols = encoder.encode(info_bits)
    
    assert symbols.size == 6
    assert np.iscomplexobj(symbols)

