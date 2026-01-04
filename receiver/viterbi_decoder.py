import numpy as np


class ViterbiDecoder:
    """
    Viterbi dekoder za konvolucijske kodove (hard-decision).

    ==========================================================
    NAČIN KORIŠTENJA
    ----------------------------------------------------------
    dec = ViterbiDecoder(
        constraint_len=7,
        generators=[0o133, 0o171, 0o164],
        rate=1/3        # informativno; stvarni broj izlaza određuju generatori
    )

    decoded_bits = dec.decode(received_bits)

    gdje je:
      - received_bits : ndarray (0/1), kodirani bitovi
      - decoded_bits  : ndarray (0/1), dekodirani informacijski bitovi
    ==========================================================

    Napomena:
    - Dekoder NE baca izuzetke za pogrešan rate ili dužinu ulaza.
    - Dekodira koliko je moguće (RX-robustno ponašanje).
    - Greške se detektuju na višem sloju (npr. CRC).
    """

    def __init__(self, constraint_len, generators, rate=1/2):
        self.K = int(constraint_len)
        self.generators = [int(g) for g in generators]
        self.rate = float(rate)  # informativno

        # STVARNA ISTINA: broj izlaznih bitova = broj generatora
        self.n_out = len(self.generators)

        # Broj stanja u trellis-u
        self.num_states = 2 ** (self.K - 1)

        # Generator tapovi (NumPy)
        self.taps = np.array(
            [
                [(g >> (self.K - 1 - i)) & 1 for i in range(self.K)]
                for g in self.generators
            ],
            dtype=np.int8
        )

        self._build_trellis()

    def _build_trellis(self):
        """
        Gradi trellis:
        - next_state[s, b]  → sljedeće stanje
        - output_bits[s, b] → izlazni bitovi
        """
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.output_bits = np.zeros((self.num_states, 2, self.n_out), dtype=np.int8)

        for state in range(self.num_states):
            # sadržaj registra (K-1 bit)
            reg = np.array(
                [(state >> i) & 1 for i in range(self.K - 2, -1, -1)],
                dtype=np.int8
            )

            for bit in (0, 1):
                v = np.concatenate(([bit], reg))
                self.output_bits[state, bit] = (self.taps @ v) % 2
                self.next_state[state, bit] = (bit << (self.K - 2)) | (state >> 1)

    def decode(self, bits_or_soft):
        """
        Hard-decision Viterbi dekodiranje.

        Parametar
        ---------
        bits_or_soft : ndarray
            Kodirani bitovi (0/1).

        Povrat
        ------
        ndarray
            Dekodirani informacijski bitovi (0/1).
        """
        rcv = np.asarray(bits_or_soft, dtype=np.int8).ravel()

        # Koliko vremenskih koraka možemo dekodirati
        T = rcv.size // self.n_out
        if T == 0:
            return np.array([], dtype=np.int8)

        # Odreži višak (robustnost)
        rcv = rcv[:T * self.n_out].reshape(T, self.n_out)

        # Path metric
        path = np.full((T + 1, self.num_states), np.inf)
        path[0, 0] = 0.0

        prev_state = np.zeros((T, self.num_states), dtype=np.int32)
        prev_bit = np.zeros((T, self.num_states), dtype=np.int8)

        # Forward pass
        for t in range(T):
            for s in range(self.num_states):
                if not np.isfinite(path[t, s]):
                    continue

                for b in (0, 1):
                    ns = self.next_state[s, b]
                    metric = path[t, s] + np.sum(
                        rcv[t] != self.output_bits[s, b]
                    )

                    if metric < path[t + 1, ns]:
                        path[t + 1, ns] = metric
                        prev_state[t, ns] = s
                        prev_bit[t, ns] = b

        # Traceback
        decoded = np.zeros(T, dtype=np.int8)
        state = np.argmin(path[T])

        for t in range(T - 1, -1, -1):
            decoded[t] = prev_bit[t, state]
            state = prev_state[t, state]

        return decoded


# ------------------------------------------------------------
# SELF-TEST (opcionalno)
# ------------------------------------------------------------
if __name__ == "__main__":
    from transmitter.convolutional import ConvolutionalEncoder

    u = np.random.randint(0, 2, 20, dtype=np.uint8)

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

    decoded = dec.decode(coded)
    print("Round-trip OK:", np.array_equal(u, decoded))
