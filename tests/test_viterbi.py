import numpy as np

from transmitter.convolutional import ConvolutionalEncoder
from receiver.viterbi_decoder import ViterbiDecoder


def test_conv_viterbi_roundtrip():
    """
    Round-trip test:
    convolutional encoder (TX) -> Viterbi decoder (RX)

    Ako ovo prođe, TX i RX su matematički kompatibilni.
    """
    u = np.random.randint(0, 2, 50, dtype=np.uint8)

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
