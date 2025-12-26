import numpy as np


class ViterbiDecoder:
    """
    Viterbi dekoder za konvolucijske kodove (hard-decision).

    Parametri
    ---------
    constraint_len : int
        Constraint length K.
    generators : list[int]
        Generator polinomi u oktalnom obliku (npr. [0o133, 0o171, 0o164]).
    rate : float
        Kodna stopa (npr. 1/3).

    Primjer
    -------
    >>> import numpy as np
    >>> from transmitter.convolutional import ConvolutionalEncoder
    >>> from receiver.viterbi_decoder import ViterbiDecoder
    >>>
    >>> u = np.random.randint(0, 2, 20, dtype=np.uint8)
    >>> enc = ConvolutionalEncoder(
    ...     constraint_len=7,
    ...     generators_octal=(0o133, 0o171, 0o164),
    ...     tail_biting=False
    ... )
    >>> coded = enc.encode(u)
    >>>
    >>> dec = ViterbiDecoder(
    ...     constraint_len=7,
    ...     generators=[0o133, 0o171, 0o164],
    ...     rate=1/3
    ... )
    >>> u_hat = dec.decode(coded)
    >>> np.array_equal(u, u_hat)
    True
    """

    def __init__(self, constraint_len, generators, rate=1/3):
        self.K = int(constraint_len)
        self.generators = [int(g) for g in generators]
        self.rate = rate

        self.num_states = 2 ** (self.K - 1)

        # Pretvaranje generatora u binarne tapove
        self.taps = [
            np.array([(g >> (self.K - 1 - i)) & 1 for i in range(self.K)], dtype=int)
            for g in self.generators
        ]

        self.next_state = {}
        self.output_bits = {}
        self._build_trellis()

    def _build_trellis(self):
        """Gradi trellis: (state, input_bit) -> next_state, output_bits"""
        for state in range(self.num_states):
            reg = np.array(
                [(state >> i) & 1 for i in range(self.K - 2, -1, -1)],
                dtype=int
            )
            for bit in (0, 1):
                v = np.concatenate(([bit], reg))
                outputs = [(np.sum(v * t) % 2) for t in self.taps]
                next_state = ((bit << (self.K - 2)) | (state >> 1))
                self.next_state[(state, bit)] = next_state
                self.output_bits[(state, bit)] = np.array(outputs, dtype=int)

    def decode(self, received_bits):
        """
        Hard-decision Viterbi dekodiranje.

        Parametar
        ---------
        received_bits : ndarray
            Niz primljenih bitova (0/1).

        Povrat
        ------
        ndarray
            Dekodirani ulazni bitovi.
        """
        rcv = np.asarray(received_bits, dtype=int).reshape(-1)

        # broj izlaznih bitova po ulaznom bitu
        n_out = int(1 / self.rate)

        # validacija dužine
        if rcv.size % n_out != 0:
            raise ValueError(
                f"Input length {rcv.size} not divisible by n_out={n_out}"
            )

        T = rcv.size // n_out

        path = np.full((T + 1, self.num_states), np.inf)
        path[0, 0] = 0  # početno stanje = 0

        prev = np.zeros((T, self.num_states), dtype=int)
        bit_dec = np.zeros((T, self.num_states), dtype=int)

        # Forward pass
        for t in range(T):
            r = rcv[t * n_out:(t + 1) * n_out]
            for s in range(self.num_states):
                if not np.isfinite(path[t, s]):
                    continue
                for b in (0, 1):
                    ns = self.next_state[(s, b)]
                    out = self.output_bits[(s, b)]
                    m = path[t, s] + np.sum(r != out)
                    if m < path[t + 1, ns]:
                        path[t + 1, ns] = m
                        prev[t, ns] = s
                        bit_dec[t, ns] = b

        # Traceback
        s = np.argmin(path[T])
        decoded = []
        for t in range(T - 1, -1, -1):
            decoded.append(bit_dec[t, s])
            s = prev[t, s]

        return np.array(decoded[::-1], dtype=int)


# Brzi self-test
if __name__ == "__main__":
    from transmitter.convolutional import ConvolutionalEncoder

    u = np.random.randint(0, 2, 30, dtype=np.uint8)
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

    print("Round-trip OK:", np.array_equal(u, u_hat))
