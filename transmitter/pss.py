import numpy as np

class PSSGenerator:
    """
    Klasa za generisanje LTE Primary Synchronization Signal (PSS) sekvence.

    PSS sekvenca je definisana u 3GPP TS 36.211, sekcija 6.11.
    Sekvenca je Zadoff-Chu (ZC) sekvenca dužine 62 sa različitim korijenom u
    zavisnosti od N_ID_2 (0, 1 ili 2).

    Metode
    -------
    generate(n_id_2):
        Statička metoda. Generiše kompleksnu PSS sekvencu dužine 62.
    """

    # Mapping prema 3GPP 36.211 Table 6.11.1.1-1
    _u_map = {
        0: 25,
        1: 29,
        2: 34
    }

    @staticmethod
    def generate(n_id_2: int) -> np.ndarray:
        """
        Generiše LTE PSS Zadoff–Chu sekvencu dužine 62.

        Parametri
        ----------
        n_id_2 : int
            Može biti samo {0, 1, 2}. Određuje korijen ZC sekvence.

        Povratna vrijednost
        -------------------
        np.ndarray (complex64)
            Numpy vektor dužine 62 koji predstavlja PSS sekvencu.

        Primjer
        -------
        >>> pss = PSSGenerator.generate(0)
        >>> print(pss.shape)
        (62,)
        """

        if n_id_2 not in PSSGenerator._u_map:
            raise ValueError("n_id_2 mora biti 0, 1 ili 2.")

        u = PSSGenerator._u_map[n_id_2]

        d = np.zeros(62, dtype=complex)

        # n = 0..30
        n1 = np.arange(0, 31)
        d[n1] = np.exp(-1j * np.pi * u * n1 * (n1 + 1) / 63)

        # n = 31..61
        n2 = np.arange(31, 62)
        d[n2] = np.exp(-1j * np.pi * u * (n2 + 1) * (n2 + 2) / 63)

        return d


# ----------------------------------------------------------
# WRAPPER (optional, za kompatibilnost sa starim pozivima)
# ----------------------------------------------------------
def generate_pss_sequence(n_id_2):
    """
    Wrapper funkcija radi kompatibilnosti sa starim kodom.

    Preporučeno je koristiti:
        PSSGenerator.generate(n_id_2)
    """
    return PSSGenerator.generate(n_id_2)