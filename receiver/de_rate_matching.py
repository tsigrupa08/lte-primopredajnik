import numpy as np

class DeRateMatcher:
    """
    DeRateMatcher vrši de-rate-matching primljenih bitova u LTE prijemniku.
    Nakon de-rate-matchinga, bitovi se vraćaju u originalni raspored
    prije rate matchinga.

    Namjena:
        - Vraćanje primljenih bitova u originalni raspored
        - Akumulacija soft ili hard bitova primljenih nakon rate matchinga

    Korištenje:
        - Instancira se sa energijom primljenih bitova (E_rx) i brojem
          originalno kodiranih bitova (N_coded)
        - Nad instancom se poziva metoda `accumulate` sa primljenim bitovima

    Attributes:
        E_rx (float): Energija primljenih bitova (samo se čuva).
        N_coded (int): Broj originalno kodiranih bitova prije rate matchinga.
    """

    def __init__(self, E_rx: float, N_coded: int):
        self.E_rx = E_rx
        self.N_coded = N_coded

    def accumulate(self, bits_rx, soft: bool = True) -> np.ndarray:
        """
        De-rate-matching: akumulacija primljenih bitova i vraćanje
        u originalni raspored dužine N_coded.

        Args:
            bits_rx (array-like): Primljeni bitovi nakon rate matchinga
                                  (soft vrijednosti ili 0/1 hard bitovi).
            soft (bool): Ako True, vraća prosječne soft vrijednosti;
                         ako False, vraća hard decision bitove (0 ili 1).

        Returns:
            np.ndarray: Niz dužine N_coded sa akumuliranim bitovima.

        Korištenje:
            - Metoda se poziva nad instancom DeRateMatcher klase
            - Prima primljene bitove i opciju soft/hard
            - Vraća niz bitova vraćen u originalni raspored
        """
        bits_rx = np.asarray(bits_rx, dtype=float)

        # Indeksi za mapiranje primljenih bitova na originalne pozicije
        indices = np.arange(bits_rx.size) % self.N_coded

        # Vektorizirana akumulacija i broj pojavljivanja po pozicijama
        weighted_sum = np.bincount(indices, weights=bits_rx, minlength=self.N_coded)
        counts = np.bincount(indices, minlength=self.N_coded)

        # Sprečavanje deljenja sa nulom
        counts[counts == 0] = 1

        # Prosječne vrijednosti po pozicijama
        soft_bits = weighted_sum / counts

        return soft_bits if soft else (soft_bits >= 0.5).astype(int)
