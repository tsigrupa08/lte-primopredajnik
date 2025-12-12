import numpy as np
import pytest
from transmitter.pbch import PBCHEncoder

# ================================================================
#                       HAPPY PATH TESTOVI
# ================================================================

def test_happy_path_basic_encode():
    """
    Happy path test kompletne enkodiranja:
    - ulaz 24 random bita
    - provjerava se da encode() vraća 960 QPSK simbola
    - provjerava se tip (kompleksni)
    """
    enc = PBCHEncoder(verbose=False)
    info_bits = np.random.randint(0, 2, 24)
    syms = enc.encode(info_bits)
    assert len(syms) == 960
    assert np.iscomplexobj(syms)


def test_happy_path_crc_fec_rate_match_qpsk():
    """
    Happy path provjera pojedinačnih koraka:
    - CRC16 (24->40)
    - FEC 1/3 (40->120)
    - Rate matching (120->1920)
    - QPSK (1920->960)
    """
    enc = PBCHEncoder(verbose=False)
    info_bits = np.random.randint(0, 2, 24)

    crc = enc.crc16(info_bits)
    assert len(crc) == 16
    bits_40 = np.concatenate((info_bits, crc))
    assert len(bits_40) == 40

    bits_120 = enc.fec_one_third(bits_40)
    assert len(bits_120) == 120

    bits_1920 = enc.rate_match(bits_120, E=1920)
    assert len(bits_1920) == 1920

    syms = enc.qpsk(bits_1920)
    assert len(syms) == 960
    assert np.iscomplexobj(syms)


def test_happy_path_small_known_input():
    """
    Happy path sa poznatim malim ulazom:
    - provjerava da encode() radi i vraća 960 QPSK simbola
    """
    enc = PBCHEncoder(verbose=False)
    info_bits = np.array([0,1]*12)  # 24 bita
    syms = enc.encode(info_bits)
    assert len(syms) == 960
    assert np.iscomplexobj(syms)


# ================================================================
#                       DUMMY TESTOVI
# ================================================================

def test_dummy_crc16_output():
    """
    Dummy test za CRC16 funkciju:
    - provjerava da vraća 16 bita
    - svi bitovi su 0 ili 1
    """
    enc = PBCHEncoder(verbose=False)
    bits = np.array([1,0,1,1])
    crc = enc.crc16(bits)
    assert len(crc) == 16
    assert np.all(np.isin(crc, [0,1]))


def test_dummy_fec_one_third_output():
    """
    Dummy test za FEC rate 1/3:
    - ulazni niz od 4 bita -> izlaz 12 bita
    """
    enc = PBCHEncoder(verbose=False)
    bits = np.array([1,0,1,0])
    fec = enc.fec_one_third(bits)
    assert len(fec) == 12
    assert np.all(np.isin(fec, [0,1]))


def test_dummy_rate_matching_repeat():
    """
    Dummy test za rate matching:
    - ulaz kraći od ciljnog -> ponavljanje
    """
    enc = PBCHEncoder(verbose=False)
    bits = np.array([0,1,1])
    rm = enc.rate_match(bits, E=9)
    expected = np.tile(bits, 3)
    assert np.array_equal(rm, expected)


def test_dummy_rate_matching_truncate():
    """
    Dummy test za rate matching:
    - ulaz duži od ciljnog -> skraćivanje
    """
    enc = PBCHEncoder(verbose=False)
    bits = np.arange(12) % 2
    rm = enc.rate_match(bits, E=5)
    assert len(rm) == 5
    assert np.all(np.isin(rm, [0,1]))


def test_dummy_qpsk_map_even_length():
    """
    Dummy test za QPSK mapiranje:
    - ulaz parnog broja bitova
    - izlaz kompleksni simboli
    """
    enc = PBCHEncoder(verbose=False)
    bits = np.array([0,1,1,0])
    syms = enc.qpsk(bits)
    assert len(syms) == 2
    assert np.iscomplexobj(syms)


def test_dummy_qpsk_map_odd_length():
    """
    Dummy test za QPSK mapiranje:
    - ulaz neparnog broja bitova -> encoder dodaje 0
    """
    enc = PBCHEncoder(verbose=False)
    bits = np.array([1,0,1])
    syms = enc.qpsk(bits)
    assert len(syms) == 2
    assert np.iscomplexobj(syms)


# ================================================================
#                     FULL CHAIN DUMMY TEST
# ================================================================

def test_dummy_full_chain_small_target():
    """
    Dummy test cijelog PBCH lanca sa malim inputom:
    - provjerava da encode() vraća 960 QPSK simbola
    """
    enc = PBCHEncoder(verbose=False)
    info_bits = np.array([1,0]*12)
    syms = enc.encode(info_bits)
    assert len(syms) == 960
    assert np.iscomplexobj(syms)
