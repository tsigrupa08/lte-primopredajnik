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
# PRIMJERI KORIŠTENJA (NumPy stil – dokumentacijski primjeri)
# ==============================================================

# --------------------------------------------------------------
# Primjer 1: Idealni QPSK simboli (bez šuma)
# --------------------------------------------------------------
# QPSK simboli (I + jQ) i pripadajući bitovi:
#   1 + 1j   → 00
#  -1 + 1j   → 10
#   1 - 1j   → 01
#  -1 - 1j   → 11

symbols_ideal = np.array([
    1 + 1j,
   -1 + 1j,
    1 - 1j,
   -1 - 1j
], dtype=np.complex64)

demapper = QPSKDemapper(mode="hard")
bits_ideal = demapper.demap(symbols_ideal)

# Očekivani rezultat:
# bits_ideal = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8)


# --------------------------------------------------------------
# Primjer 2: QPSK simboli s malim šumom
# --------------------------------------------------------------
# Hard decision demapper koristi samo predznak I i Q komponente.

symbols_noisy = np.array([
    0.8 + 1.2j,
   -1.1 + 0.9j,
    0.7 - 1.3j,
   -0.6 - 0.8j
], dtype=np.complex64)

bits_noisy = demapper.demap(symbols_noisy)

# Očekivani rezultat (isti kao idealni slučaj):
# bits_noisy = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8)


# --------------------------------------------------------------
# Primjer 3: Jedan QPSK simbol
# --------------------------------------------------------------
# Ulaz može biti i niz duljine 1.

single_symbol = np.array([1 - 1j], dtype=np.complex64)
bits_single = demapper.demap(single_symbol)

# Očekivani rezultat:
# bits_single = np.array([0, 1], dtype=np.uint8)


# --------------------------------------------------------------
# Primjer 4: Veći broj simbola (vektorizirana obrada)
# --------------------------------------------------------------
# Demapper je u potpunosti vektoriziran i radi nad cijelim nizom.

symbols_many = np.random.choice(
    [1+1j, -1+1j, 1-1j, -1-1j],
    size=100
).astype(np.complex64)

bits_many = demapper.demap(symbols_many)

# Napomena:
# - Duzina izlaza je 2 * len(symbols_many)
# - bits[0::2] → bitovi iz I-komponente (b0)
# - bits[1::2] → bitovi iz Q-komponente (b1)
