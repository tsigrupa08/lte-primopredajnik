# rx/viterbi_decoder.py

import numpy as np


class ViterbiDecoder:
    """
    Viterbi decoder for convolutional codes (hard-decision).

    Parameters
    ----------
    constraint_len : int
        Constraint length (K).
    generators : list[int]
        Generator polynomials in octal or binary form.
    rate : float
        Code rate (e.g. 1/2 or 1/3).
    """

    def __init__(self, constraint_len, generators, rate=1/3):
        self.K = constraint_len
        self.generators = generators
        self.rate = rate

        self.num_states = 2 ** (self.K - 1)

        # Precompute trellis
        self.next_state = {}
        self.output_bits = {}
        self._build_trellis()

    def _build_trellis(self):
        """
        Builds trellis:
        (state, input_bit) -> next_state, output_bits
        """
        for state in range(self.num_states):
            for bit in (0, 1):
                shift_reg = ((bit << (self.K - 1)) | state)

                outputs = []
                for g in self.generators:
                    masked = shift_reg & g
                    outputs.append(bin(masked).count("1") % 2)

                next_state = shift_reg >> 1

                self.next_state[(state, bit)] = next_state
                self.output_bits[(state, bit)] = np.array(outputs)

    def decode(self, received_bits):
        """
        Hard-decision Viterbi decoding.

        Parameters
        ----------
        received_bits : ndarray
            1D array of received bits (0/1), length = rate * N

        Returns
        -------
        decoded_bits : ndarray
            Estimated input bit sequence.
        """
        received_bits = np.array(received_bits, dtype=int)
        n_outputs = len(self.generators)
        num_steps = len(received_bits) // n_outputs

        path_metric = np.full((num_steps + 1, self.num_states), np.inf)
        path_metric[0, 0] = 0  # start from zero state

        predecessor = np.zeros((num_steps, self.num_states), dtype=int)
        decided_bit = np.zeros((num_steps, self.num_states), dtype=int)

        for t in range(num_steps):
            r = received_bits[t * n_outputs:(t + 1) * n_outputs]

            for state in range(self.num_states):
                if path_metric[t, state] == np.inf:
                    continue

                for bit in (0, 1):
                    ns = self.next_state[(state, bit)]
                    out = self.output_bits[(state, bit)]

                    metric = np.sum(r != out)
                    new_metric = path_metric[t, state] + metric

                    if new_metric < path_metric[t + 1, ns]:
                        path_metric[t + 1, ns] = new_metric
                        predecessor[t, ns] = state
                        decided_bit[t, ns] = bit

        # Traceback
        state = np.argmin(path_metric[num_steps])
        decoded = []

        for t in reversed(range(num_steps)):
            decoded.append(decided_bit[t, state])
            state = predecessor[t, state]

        return np.array(decoded[::-1])
