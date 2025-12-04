import numpy as np
import pytest
from transmitter.pss import PSSGenerator, generate_pss_sequence


# ----------------------------------------------------------
# BASIC HAPPY-PATH TESTS
# ----------------------------------------------------------

def test_pss_length():
    """Svaka PSS sekvenca mora imati tačno 62 elementa."""
    for nid2 in [0, 1, 2]:
        seq = PSSGenerator.generate(nid2)
        assert len(seq) == 62


def test_pss_type_complex():
    """Sekvenca mora biti tipa kompleksnih brojeva."""
    for nid2 in [0, 1, 2]:
        seq = PSSGenerator.generate(nid2)
        assert np.issubdtype(seq.dtype, np.complexfloating)


def test_pss_wrapper_equivalence():
    """Wrapper funkcija mora davati istu sekvencu kao OOP metoda."""
    for nid2 in [0, 1, 2]:
        seq1 = PSSGenerator.generate(nid2)
        seq2 = generate_pss_sequence(nid2)
        assert np.allclose(seq1, seq2)


def test_pss_uniqueness_between_root_indices():
    """PSS sekvence za različite N_ID_2 moraju biti različite."""
    seq0 = PSSGenerator.generate(0)
    seq1 = PSSGenerator.generate(1)
    seq2 = PSSGenerator.generate(2)

    assert not np.array_equal(seq0, seq1)
    assert not np.array_equal(seq0, seq2)
    assert not np.array_equal(seq1, seq2)


# ----------------------------------------------------------
# ZC SEQUENCE PROPERTIES
# ----------------------------------------------------------

def test_pss_constant_amplitude():
    """ZC sekvenca ima konstantnu amplitudu (|x| = 1)."""
    for nid2 in [0, 1, 2]:
        seq = PSSGenerator.generate(nid2)
        magnitudes = np.abs(seq)
        assert np.allclose(magnitudes, 1.0, atol=1e-6)


def test_pss_phase_values_not_nan():
    """Fazne vrijednosti moraju biti validne (bez NaN)."""
    for nid2 in [0, 1, 2]:
        seq = PSSGenerator.generate(nid2)
        assert not np.isnan(np.angle(seq)).any()


def test_pss_sequence_non_zero():
    """Nijedan element sekvence ne smije biti nula."""
    for nid2 in [0, 1, 2]:
        seq = PSSGenerator.generate(nid2)
        assert not np.any(seq == 0)


def test_pss_conjugate_properties():
    """Provjera da conj(conj(x)) == x (osnovna stabilnost)."""
    for nid2 in [0, 1, 2]:
        seq = PSSGenerator.generate(nid2)
        assert np.allclose(np.conj(np.conj(seq)), seq)


# ----------------------------------------------------------
# ERROR HANDLING (UNHAPPY PATH)
# ----------------------------------------------------------

def test_invalid_nid2_negative():
    with pytest.raises(ValueError):
        PSSGenerator.generate(-1)


def test_invalid_nid2_too_large():
    with pytest.raises(ValueError):
        PSSGenerator.generate(3)


def test_invalid_nid2_float():
    with pytest.raises(ValueError):
        PSSGenerator.generate(1.5)


def test_invalid_nid2_string():
    with pytest.raises(ValueError):
        PSSGenerator.generate("0")


def test_invalid_nid2_none():
    with pytest.raises(ValueError):
        PSSGenerator.generate(None)


# ----------------------------------------------------------
# CONSISTENCY TESTS
# ----------------------------------------------------------

def test_pss_repeatability():
    """Dva poziva za isti nid2 moraju dati istu sekvencu."""
    for nid2 in [0, 1, 2]:
        seq1 = PSSGenerator.generate(nid2)
        seq2 = PSSGenerator.generate(nid2)
        assert np.allclose(seq1, seq2)


def test_pss_dtype_stability():
    """Provjera da dtype ostaje complex128 ili complex64."""
    for nid2 in [0, 1, 2]:
        seq = PSSGenerator.generate(nid2)
        assert seq.dtype in (np.complex128, np.complex64)