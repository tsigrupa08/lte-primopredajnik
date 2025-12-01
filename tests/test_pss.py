import numpy as np
import pytest
from transmitter.pss import generate_pss_sequence

def test_pss_length():
    """Provjerava da sekvenca ima tačno 62 elementa"""
    for nid2 in [0, 1, 2]:
        seq = generate_pss_sequence(nid2)
        assert len(seq) == 62

def test_pss_type():
    """Provjerava da je sekvenca tipa kompleksnih brojeva"""
    for nid2 in [0, 1, 2]:
        seq = generate_pss_sequence(nid2)
        assert np.issubdtype(seq.dtype, np.complexfloating)

def test_pss_uniqueness():
    """Provjerava da su PSS sekvence različite za različite nid2"""
    seq0 = generate_pss_sequence(0)
    seq1 = generate_pss_sequence(1)
    seq2 = generate_pss_sequence(2)
    
    # sve tri sekvence moraju biti različite
    assert not np.array_equal(seq0, seq1)
    assert not np.array_equal(seq0, seq2)
    assert not np.array_equal(seq1, seq2)
