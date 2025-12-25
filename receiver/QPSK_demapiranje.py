import numpy as np


class QPSKDemapper:
    """
    QPSK Demapper (hard decision)

    Inverzni korak QPSK mapiranja korištenog u PBCHEncoder.qpsk().

    Mapiranje u encoderu:
        I = 1 - 2*b0
        Q = 1 - 2*b1

    Inverzno (hard decision):
        b0 = 0 ako I > 0, inače 1
        b1 = 0 ako Q > 0, inače 1
    """

    def __init__(self, mode="hard"):
        """
        Parametri
        ----------
        mode : str
            Trenutno podržan samo "hard" (hard-decision demapping)
        """
        if mode != "hard":
            raise NotImplementedError(
                "Trenutno je podržan samo hard demapping"
            )

        self.mode = mode

    # ==============================================================
    # QPSK DEMAPIRANJE
    # ==============================================================

    def demap(self, symbols):
        """
        Demapira QPSK simbole u bitove.

        Parametri
        ----------
        symbols : numpy.ndarray of complex
            QPSK simboli (npr. izlaz OFDM demodulatora nakon equalizacije)

        Povrat
        -------
        bits : numpy.ndarray of uint8
            Demapirani bitovi (0/1), duljine 2 * broj_simbola
        """

        # Osiguraj numpy kompleksni tip
        symbols = np.asarray(symbols, dtype=np.complex64)

        # --------------------------------------------------
        # 1. I i Q komponente
        # --------------------------------------------------
        I = np.real(symbols)
        Q = np.imag(symbols)

        # --------------------------------------------------
        # 2. Hard decision prema PBCH QPSK mapiranju
        #
        # Encoder:
        #   I = +1 → bit 0
        #   I = -1 → bit 1
        #
        # Dakle:
        #   bit = 0 ako je komponenta > 0
        #   bit = 1 ako je komponenta <= 0
        # --------------------------------------------------
        b0 = (I <= 0).astype(np.uint8)
        b1 = (Q <= 0).astype(np.uint8)

        # --------------------------------------------------
        # 3. Spajanje bitova u izlazni niz
        #    (svaki simbol → 2 bita)
        # --------------------------------------------------
        bits = np.empty(2 * len(symbols), dtype=np.uint8)
        bits[0::2] = b0
        bits[1::2] = b1

        return bits


# ==============================================================
# PRIMJERI KORIŠTENJA (samo komentari – dokumentacija)
# ==============================================================
"""
# --------------------------------------------------------------
# Primjer 1: Idealni QPSK simboli (bez šuma)
# --------------------------------------------------------------
# symbols = np.array([
#     1 + 1j,
#    -1 + 1j,
#     1 - 1j,
#    -1 - 1j
# ], dtype=np.complex64)
#
# demapper = QPSKDemapper(mode="hard")
# bits = demapper.demap(symbols)
#
# Očekivani rezultat:
# bits = [0, 0, 1, 0, 0, 1, 1, 1]


# --------------------------------------------------------------
# Primjer 2: QPSK simboli s malim šumom
# --------------------------------------------------------------
# Hard decision demapper koristi samo predznak I i Q komponente.
#
# symbols = np.array([
#     0.8 + 1.2j,
#    -1.1 + 0.9j,
#     0.7 - 1.3j,
#    -0.6 - 0.8j
# ], dtype=np.complex64)
#
# bits = demapper.demap(symbols)
#
# Očekivani rezultat (isti kao idealni slučaj):
# bits = [0, 0, 1, 0, 0, 1, 1, 1]


# --------------------------------------------------------------
# Primjer 3: Jedan QPSK simbol
# --------------------------------------------------------------
# symbols = np.array([1 - 1j], dtype=np.complex64)
# bits = demapper.demap(symbols)
#
# Očekivani rezultat:
# bits = [0, 1]


# --------------------------------------------------------------
# Primjer 4: Veći broj simbola (vektorizirana obrada)
# --------------------------------------------------------------
# symbols = np.random.choice(
#     [1+1j, -1+1j, 1-1j, -1-1j],
#     size=100
# ).astype(np.complex64)
#
# bits = demapper.demap(symbols)
#
# Napomena:
# - Duljina izlaza je 2 * len(symbols)
# - bits[0::2] → bitovi iz I-komponente (b0)
# - bits[1::2] → bitovi iz Q-komponente (b1)
"""