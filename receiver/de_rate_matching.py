import numpy as np

class DeRateMatcher:
    """
    De-rate matcher za LTE prijemnik.

    Vrši de-rate-matching (akumulaciju) primljenih bitova nakon
    rate matchinga i vraća ih u originalni raspored prije kodiranja.

    Tipično se koristi u PBCH prijemnom lancu nakon QPSK demapiranja
    i prije Viterbi dekodiranja.

    Atributi
    --------
    E_rx : float
        Energija primljenih bitova (informativni parametar).
    N_coded : int
        Broj originalno kodiranih bitova prije rate matchinga.

    Primjeri korištenja
    ------------------
    >>> import numpy as np
    >>> from rx.derate_matcher import DeRateMatcher
    >>>
    >>> # Generišemo primljene bitove nakon rate matchinga (0 ili 1)
    >>> bits_rx = np.random.randint(0, 2, 576)
    >>> # Kreiramo objekat de-rate matchera sa energijom i brojem kodiranih bitova
    >>> drm = DeRateMatcher(E_rx=864, N_coded=192)
    >>> # Vraćamo de-rate-matched bitove (hard decision)
    >>> out = drm.accumulate(bits_rx, soft=False)
    >>> out.shape
    (192,)
    >>>
    >>> # --- PRAKTIČAN PRIMJER ---
    >>> # Simuliramo primljene soft vrijednosti bitova između 0 i 1
    >>> bits_soft = np.random.rand(576)
    >>> # Vraćamo soft vrijednosti (prosjek za pozicije koje se ponavljaju)
    >>> soft_bits = drm.accumulate(bits_soft, soft=True)
    >>> # Vraćamo hard decision bitove
    >>> hard_bits = drm.accumulate(bits_soft, soft=False)
    >>> # Prikaz prvih 10 bitova za provjeru
    >>> print("Soft bitovi:", soft_bits[:10])
    >>> print("Hard bitovi:", hard_bits[:10])

    Napomene
    --------
    - Ako se ista kodirana pozicija pojavi više puta, vrijednosti
      se akumuliraju i prosječe (soft kombinacija).
    - Implementacija je vektorizirana i pogodna za NumPy obradu.
    """

    def __init__(self, E_rx: float, N_coded: int):
        """
        Inicijalizuje DeRateMatcher objekat.

        Parametri
        ---------
        E_rx : float
            Energija primljenih bitova.
        N_coded : int
            Broj originalno kodiranih bitova prije rate matchinga.
        """
        self.E_rx = E_rx
        self.N_coded = N_coded

    def accumulate(self, bits_rx, soft: bool = True) -> np.ndarray:
        """
        Izvršava de-rate-matching (akumulaciju) primljenih bitova.

        Parametri
        ---------
        bits_rx : array-like
            Primljeni bitovi nakon rate matchinga
            (soft vrijednosti ili 0/1 hard bitovi).
        soft : bool, opcionalno
            Ako je True, vraća soft vrijednosti.
            Ako je False, vraća hard decision bitove (0 ili 1).

        Povratna vrijednost
        ------------------
        np.ndarray
            De-rate-matched niz bitova dužine `N_coded`.

        Komentari
        ---------
        - Pretvara ulaz u NumPy niz tipa float.
        - Ako se ista pozicija ponavlja, vrijednosti se prosječe (soft).
        - Hard decision se određuje thresholdom 0.5.
        """
        # Pretvaramo ulaz u NumPy niz tipa float
        bits_rx = np.asarray(bits_rx, dtype=float)

        # Određujemo pozicije gdje ide svaki primljeni bit
        indices = np.arange(bits_rx.size) % self.N_coded

        # Sabiramo vrijednosti koje idu na istu poziciju
        weighted_sum = np.bincount(
            indices, weights=bits_rx, minlength=self.N_coded
        )

        # Broj puta koliko je svaki indeks pokriven
        counts = np.bincount(indices, minlength=self.N_coded)
        counts[counts == 0] = 1  # Sprečavamo dijeljenje sa nulom

        # Soft vrijednosti (prosjek za pozicije koje se ponavljaju)
        soft_bits = weighted_sum / counts

        # Vraćamo soft ili hard bitove
        return soft_bits if soft else (soft_bits >= 0.5).astype(int)
