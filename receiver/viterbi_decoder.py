import numpy as np


class ViterbiDecoder:
    """
    Hard-decision Viterbi dekoder za konvolucijske kodove.

    Dizajniran da zadovolji pytest testove:
    - nikad ne skraćuje izlaz
    - dekodira koliko god može
    - ne pretpostavlja tail-biting
    - ne pretpostavlja flush bitove
    """

    def __init__(self, constraint_len, generators, rate=1/2):
        self.K = int(constraint_len)
        self.generators = [int(g) for g in generators]
        self.rate = float(rate)  # informativno

        self.n_out = len(self.generators)
        self.num_states = 2 ** (self.K - 1)

        # generator tapovi
        self.taps = np.array(
            [
                [(g >> (self.K - 1 - i)) & 1 for i in range(self.K)]
                for g in self.generators
            ],
            dtype=np.int8
        )

        self._build_trellis()

    def _build_trellis(self):
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.output_bits = np.zeros(
            (self.num_states, 2, self.n_out), dtype=np.int8
        )

        for state in range(self.num_states):
            reg = np.array(
                [(state >> i) & 1 for i in range(self.K - 2, -1, -1)],
                dtype=np.int8
            )

            for bit in (0, 1):
                v = np.concatenate(([bit], reg))
                self.output_bits[state, bit] = (self.taps @ v) % 2
                self.next_state[state, bit] = (
                    (bit << (self.K - 2)) | (state >> 1)
                )

    def decode(self, bits_or_soft):
        rcv = np.asarray(bits_or_soft, dtype=np.int8).ravel()

        # broj vremenskih koraka
        T = rcv.size // self.n_out
        if T == 0:
            return np.array([], dtype=np.int8)

        rcv = rcv[:T * self.n_out].reshape(T, self.n_out)

        # path metrics
        path = np.full((T + 1, self.num_states), np.inf)
        path[0, 0] = 0.0  # početno stanje

        prev_state = np.zeros((T, self.num_states), dtype=np.int32)
        prev_bit = np.zeros((T, self.num_states), dtype=np.int8)

        # forward pass
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

        # traceback
        decoded = np.zeros(T, dtype=np.int8)
        state = np.argmin(path[T])

        for t in range(T - 1, -1, -1):
            decoded[t] = prev_bit[t, state]
            state = prev_state[t, state]

        return decoded
