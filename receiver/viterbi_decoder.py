import numpy as np


class ViterbiDecoder:
    """
    Hard-decision Viterbi dekoder za konvolucijske kodove.

    Ova klasa implementira Viterbi algoritam za dekodiranje konvolucijski
    kodiranih bitova sa hard-odlukom (0/1). Posebno je prilagođena LTE PBCH
    lancu gdje se koristi **tail-biting** konvolucijsko kodiranje.

    LTE PBCH napomena:
    - Predajnik koristi tail-biting (nema flush bitova).
    - Početno stanje nije poznato unaprijed.
    - Dekoder traži start_state za koji važi: end_state == start_state
      uz minimalnu putnu metriku.

    Parameters
    ----------
    constraint_len : int
        Dužina ograničenja konvolucijskog koda (K).
    generators : list of int
        Generator polinomi konvolucijskog koda (npr. [0o133, 0o171, 0o165]).
    rate : float, default=1/2
        Kodna stopa (informativno, ne utiče direktno na algoritam).

    Attributes
    ----------
    K : int
        Dužina ograničenja konvolucijskog koda.
    generators : list of int
        Generator polinomi.
    rate : float
        Kodna stopa.
    n_out : int
        Broj izlaznih bitova po ulaznom bitu.
    num_states : int
        Ukupan broj stanja trellisa (2^(K-1)).
    """

    def __init__(self, constraint_len, generators, rate=1 / 2):
        self.K = int(constraint_len)
        self.generators = [int(g) for g in generators]
        self.rate = float(rate)  # informativno

        self.n_out = len(self.generators)
        self.num_states = 2 ** (self.K - 1)

        # generator tapovi (n_out x K)
        self.taps = np.array(
            [[(g >> (self.K - 1 - i)) & 1 for i in range(self.K)] for g in self.generators],
            dtype=np.int8,
        )

        self._build_trellis()

    def _build_trellis(self):
        """
        Izgrađuje trellis strukturu (prelazi stanja i izlazni bitovi).

        Metoda popunjava:
        - next_state[state, bit]
        - output_bits[state, bit, :]
        """
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.output_bits = np.zeros((self.num_states, 2, self.n_out), dtype=np.int8)

        for state in range(self.num_states):
            # reg = (K-1) bita stanja, MSB-first (najnoviji bit je MSB)
            reg = np.array([(state >> i) & 1 for i in range(self.K - 2, -1, -1)], dtype=np.int8)

            for bit in (0, 1):
                v = np.concatenate(([bit], reg))  # [u(t), state_bits...]
                self.output_bits[state, bit] = (self.taps @ v) % 2
                self.next_state[state, bit] = (bit << (self.K - 2)) | (state >> 1)

    def _run_viterbi(self, rcv: np.ndarray, start_state: int):
        """
        Izvršava jedan Viterbi algoritam za fiksno početno stanje.

        Parameters
        ----------
        rcv : numpy.ndarray
            Primljeni kodirani bitovi oblika (T, n_out).
        start_state : int
            Početno stanje trellisa.

        Returns
        -------
        path : numpy.ndarray
            Matrica putnih metrika dimenzija (T+1, num_states).
        prev_state : numpy.ndarray
            Prethodna stanja za backtracking.
        prev_bit : numpy.ndarray
            Prethodni ulazni bitovi za backtracking.
        """
        T = rcv.shape[0]

        path = np.full((T + 1, self.num_states), np.inf, dtype=np.float64)
        path[0, start_state] = 0.0

        prev_state = np.zeros((T, self.num_states), dtype=np.int32)
        prev_bit = np.zeros((T, self.num_states), dtype=np.int8)

        for t in range(T):
            for s in range(self.num_states):
                pm = path[t, s]
                if not np.isfinite(pm):
                    continue

                for b in (0, 1):
                    ns = self.next_state[s, b]
                    # hard metric: Hamming distance
                    metric = pm + np.sum(rcv[t] != self.output_bits[s, b])

                    if metric < path[t + 1, ns]:
                        path[t + 1, ns] = metric
                        prev_state[t, ns] = s
                        prev_bit[t, ns] = b

        return path, prev_state, prev_bit

    def decode(self, bits_or_soft, tail_biting: bool = False) -> np.ndarray:
        """
        Dekodira kodirane bitove.

        bits_or_soft: očekuje hard bitove 0/1 dužine T*n_out.
                      Ako dođe float, kvantizuje se pragom 0.5.
        tail_biting:  True -> tail-biting Viterbi (end_state == start_state)
                      False -> start_state=0, end_state=argmin
        """
        rcv_in = np.asarray(bits_or_soft).ravel()

        if rcv_in.size == 0:
            return np.array([], dtype=np.int8)

        # ako su soft vrijednosti (float), pretvori u hard 0/1
        if np.issubdtype(rcv_in.dtype, np.floating):
            rcv_in = (rcv_in >= 0.5).astype(np.int8)
        else:
            rcv_in = rcv_in.astype(np.int8)

        # broj vremenskih koraka
        T = rcv_in.size // self.n_out
        if T == 0:
            return np.array([], dtype=np.int8)

        rcv = rcv_in[: T * self.n_out].reshape(T, self.n_out)

        # -----------------------------
        # Tail-biting režim (PBCH)
        # -----------------------------
        if tail_biting:
            best_metric = np.inf
            best_start = 0
            best_prev_state = None
            best_prev_bit = None

            for s0 in range(self.num_states):
                path, prev_state, prev_bit = self._run_viterbi(rcv, start_state=s0)
                metric = path[T, s0]  # end_state mora biti == start_state

                if metric < best_metric:
                    best_metric = metric
                    best_start = s0
                    best_prev_state = prev_state
                    best_prev_bit = prev_bit

            # fallback (ne bi trebalo da se desi, ali radi robusnosti)
            if best_prev_state is None:
                path, best_prev_state, best_prev_bit = self._run_viterbi(rcv, start_state=0)
                end_state = int(np.argmin(path[T]))
            else:
                end_state = int(best_start)

            decoded = np.zeros(T, dtype=np.int8)
            state = end_state
            for t in range(T - 1, -1, -1):
                decoded[t] = best_prev_bit[t, state]
                state = best_prev_state[t, state]

            return decoded

        # -----------------------------
        # Standardni režim
        # -----------------------------
        path, prev_state, prev_bit = self._run_viterbi(rcv, start_state=0)

        decoded = np.zeros(T, dtype=np.int8)
        state = int(np.argmin(path[T]))

        for t in range(T - 1, -1, -1):
            decoded[t] = prev_bit[t, state]
            state = prev_state[t, state]

        return decoded
