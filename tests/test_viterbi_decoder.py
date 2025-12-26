import numpy as np
import pytest
from transmitter.convolutional import ConvolutionalEncoder
from receiver.viterbi_decoder import ViterbiDecoder


def test_viterbi_decoder_round_trip():
    """Happy path: encoder + decoder vraćaju originalni niz."""
    u = np.array([1, 0, 1, 1, 0], dtype=np.uint8)

    enc = ConvolutionalEncoder(
        constraint_len=7,
        generators_octal=(0o133, 0o171, 0o164),
        tail_biting=False
    )
    coded = enc.encode(u)

    dec = ViterbiDecoder(
        constraint_len=7,
        generators=[0o133, 0o171, 0o164],
        rate=1/3
    )
    u_hat = dec.decode(coded)

    assert np.array_equal(u, u_hat)


def test_viterbi_decoder_unhappy_length():
    """Unhappy path: ulazna dužina nije višekratnik broja izlaza → očekujemo ValueError."""
    dec = ViterbiDecoder(
        constraint_len=7,
        generators=[0o133, 0o171, 0o164],
        rate=1/3
    )
    bad_bits = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)

    with pytest.raises(ValueError):
        dec.decode(bad_bits)


def test_viterbi_decoder_unhappy_rate():
    """Unhappy path: pogrešan rate → očekujemo ValueError."""
    u = np.array([1, 0, 1, 1, 0], dtype=np.uint8)

    enc = ConvolutionalEncoder(
        constraint_len=7,
        generators_octal=(0o133, 0o171, 0o164),
        tail_biting=False
    )
    coded = enc.encode(u)

    dec = ViterbiDecoder(
        constraint_len=7,
        generators=[0o133, 0o171, 0o164],
        rate=1/2  # namjerno pogrešan
    )

    with pytest.raises(ValueError):
        dec.decode(coded)
