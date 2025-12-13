import numpy as np
import pytest

from transmitter.pss import generate_pss_sequence
from transmitter.resource_grid import create_resource_grid
from transmitter.pbch import PBCHEncoder
from transmitter.ofdm import OFDMModulator
from lte_tx_chain import LTETxChain  # prilagodi import prema tvojoj strukturi


# ---------------------------
# HAPPY TESTS
# ---------------------------

def test_init_creates_grid():
    tx = LTETxChain(nid2=1, ndlrb=6, num_subframes=1, normal_cp=True)
    assert isinstance(tx.grid, np.ndarray)
    assert tx.grid.shape[0] > 0  # grid mora imati dimenzije


def test_generate_waveform_without_mib():
    tx = LTETxChain()
    waveform, fs = tx.generate_waveform()
    assert isinstance(waveform, np.ndarray)
    assert np.iscomplexobj(waveform)  # OFDM izlaz mora biti kompleksan
    assert isinstance(fs, (int, float))
    assert fs > 0


def test_generate_waveform_with_mib():
    tx = LTETxChain()
    mib_bits = np.random.randint(0, 2, size=24)  # tipičan MIB niz
    waveform, fs = tx.generate_waveform(mib_bits=mib_bits)
    assert isinstance(waveform, np.ndarray)
    assert waveform.size > 0
    assert fs > 0


def test_pss_sequence_generation_matches_length():
    pss, nid2 = generate_pss_sequence(0)
    assert len(pss) == 62  # LTE PSS sekvenca je uvijek 62 simbola
    assert nid2 in [0, 1, 2]


# ---------------------------
# SAD TESTS
# ---------------------------

def test_invalid_nid2_raises():
    with pytest.raises(ValueError):
        LTETxChain(nid2=5)  # dozvoljeno je samo 0–2


def test_invalid_ndlrb_raises():
    with pytest.raises(ValueError):
        LTETxChain(ndlrb=3)  # minimalno je 6 RB


def test_generate_waveform_with_invalid_mib_bits():
    tx = LTETxChain()
    mib_bits = [2, 3, 4]  # nevalidni bitovi (mora biti 0 ili 1)
    with pytest.raises(ValueError):
        tx.generate_waveform(mib_bits=mib_bits)


def test_pbch_encoder_target_bits_mismatch():
    encoder = PBCHEncoder(target_bits=384)
    bad_bits = np.random.randint(0, 2, size=10)  # premalo bitova
    with pytest.raises(ValueError):
        encoder.encode(bad_bits)


def test_ofdm_modulator_invalid_grid():
    with pytest.raises(ValueError):
        OFDMModulator(grid=None).modulate()