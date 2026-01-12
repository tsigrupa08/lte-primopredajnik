# transmitter/convolutional.py

import numpy as np
from typing import Sequence


class ConvolutionalEncoder:
    """
    LTE convolutional encoder (rate 1/3, hard-decision)

    Constraint length:
        K = 7

    Generator polynomials (octal):
        [133, 171, 165]
    """

    def __init__(
        self,
        constraint_len: int = 7,
        generators_octal: Sequence[int] = (0o133, 0o171, 0o165),
        tail_biting: bool = False,
    ):
        self.K = constraint_len
        self.generators = np.asarray(generators_octal, dtype=int)
        self.n_out = self.generators.size
        self.tail_biting = tail_biting

        # --------------------------------------------------
        # Generator taps matrix
        # shape: (n_out, K)
        # --------------------------------------------------
        self.taps = np.zeros((self.n_out, self.K), dtype=np.uint8)
        for i, g in enumerate(self.generators):
            # MSB -> LSB
            self.taps[i] = [(g >> (self.K - 1 - k)) & 1 for k in range(self.K)]

    def encode(self, bits: Sequence[int]) -> np.ndarray:
        """
        Convolutional encoding (hard bits).

        Parameters
        ----------
        bits : Sequence[int]
            Input bits (0/1), shape (N,)

        Returns
        -------
        np.ndarray
            Encoded bits, shape (N * n_out,)
        """
        
        u = np.asarray(bits, dtype=np.uint8).reshape(-1)
        N = u.size

        # --- INIT SHIFT REGISTAR ---
        reg = np.zeros(self.K - 1, dtype=np.uint8)

        # Tail-biting: početno stanje = zadnjih (K-1) bitova ulaza
        if self.tail_biting and N > 0:
            # reg[0] = u[N-1], reg[1] = u[N-2], ...
            reg = np.take(u, np.arange(-1, -self.K, -1), mode="wrap").astype(np.uint8)

        out = np.zeros((N, self.n_out), dtype=np.uint8)

        for i in range(N):
            state = np.empty(self.K, dtype=np.uint8)
            state[0] = u[i]
            state[1:] = reg

            out[i] = (self.taps @ state) & 1

            reg[1:] = reg[:-1]
            reg[0] = u[i]

        return out.reshape(-1)


# --------------------------------------------------
# enc = ConvolutionalEncoder(
#     constraint_len=7,
#     generators_octal=(0o133, 0o171, 0o165),
#     tail_biting=False
# )
#
# u = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
# y = enc.encode(u)
#
# # y length = len(u) * 3 (rate 1/3)
# --------------------------------------------------


if __name__ == "__main__":
    # Simple self-test
    u = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    enc = ConvolutionalEncoder()
    y = enc.encode(u)

    print("Convolutional Encoder Test")
    print("Input :", u)
    print("Output:", y)
    print("Length:", y.size)
