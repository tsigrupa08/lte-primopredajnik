"""
PBCHEncoder
===========

Ova klasa implementira pojednostavljeni PBCH enkoder prema traženoj
specifikaciji projektne zadaće:

    24 → 40 → 120 → 1920 → 960

Gdje je:
- 24  = informacijski bitovi
- 40  = informacijski + CRC16
- 120 = FEC kodirani bitovi (rate 1/3)
- 1920 = nakon rate-matchinga
- 960 = broj QPSK simbola

Implementirane komponente:
- CRC16 (polinom 0x1021)
- FEC rate 1/3 (jednostavan sistematski + 2 pariteta)
- Rate matching (repeat/prune)
- QPSK mapiranje (Gray)

Ovo NIJE 3GPP-standardni PBCH, nego pojednostavljena instrukcijska
implementacija za potrebe projektne vježbe.
"""

from typing import Sequence
import numpy as np
import math


class PBCHEncoder:
    """
    PBCHEncoder: kompletan enkoder za PBCH zadat iz projektne dokumentacije.

    Parameters
    ----------
    verbose : bool, optional
        Ako je True, ispisuje informacije o svakom koraku enkodiranja.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # =====================================================================
    #                           CRC16
    # =====================================================================
    def crc16(self, bits: Sequence[int]) -> np.ndarray:
        """
        Izračunava CRC16 remainder (16 bita) nad ulaznim bitovima.

        CRC polinom: 0x1021 (CRC-16-IBM)

        Parameters
        ----------
        bits : Sequence[int]
            Ulazni bitovi (0/1).

        Returns
        -------
        numpy.ndarray
            CRC remainder dužine 16 bitova (np.uint8).
        """
        poly = 0x1021
        reg = 0xFFFF

        for b in bits:
            reg ^= (int(b) & 1) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF

        return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)

    # =====================================================================
    #                      RATE 1/3 FEC ENCODER
    # =====================================================================
    def fec_one_third(self, bits: Sequence[int]) -> np.ndarray:
        """
        Jednostavni FEC rate 1/3 encoder:
        - systematic bit,
        - parity1 = b XOR 1,
        - parity2 = b (kopija)

        Ovo daje ukupno 3 output bita za svaki input bit.

        Parameters
        ----------
        bits : Sequence[int]
            Ulazni bitovi dužine 40.

        Returns
        -------
        numpy.ndarray
            Kodirani bitovi dužine 120.
        """
        b = np.array(bits, dtype=np.uint8)
        out = []

        for x in b:
            out.append(x)       # systematic
            out.append(x ^ 1)   # parity1
            out.append(x)       # parity2

        return np.array(out, dtype=np.uint8)

    # =====================================================================
    #                       RATE MATCHING
    # =====================================================================
    def rate_match(self, bits: Sequence[int], E: int = 1920) -> np.ndarray:
        """
        Rate matching prema ciljanom broju bitova.

        Ako je ulaz kraći od E → ponavlja se (tile).
        Ako je ulaz duži od E → poduzima se linearni sampling.

        Parameters
        ----------
        bits : Sequence[int]
            Kodirani bitovi (npr. 120).
        E : int
            Ciljni broj bitova (1920).

        Returns
        -------
        numpy.ndarray
            Rate-matched niz dužine E.
        """
        b = np.array(bits, dtype=np.uint8)
        N = len(b)

        if N == E:
            return b.copy()

        if N < E:
            reps = math.ceil(E / N)
            big = np.tile(b, reps)
            return big[:E]

        idx = np.linspace(0, N - 1, num=E).round().astype(int)
        return b[idx]

    # =====================================================================
    #                       QPSK MAPPING
    # =====================================================================
    def qpsk(self, bits: Sequence[int]) -> np.ndarray:
        """
        Gray QPSK mapiranje:

        00 → +1 + j1
        01 → +1 - j1
        11 → -1 - j1
        10 → -1 + j1

        Pravougaoni mapping:
            I = 1 - 2*b0
            Q = 1 - 2*b1

        Parameters
        ----------
        bits : Sequence[int]
            Ulažni bitovi dužine 1920.

        Returns
        -------
        numpy.ndarray
            QPSK simboli (960 kompleksnih vrijednosti).
        """
        b = np.array(bits, dtype=np.uint8)

        if len(b) % 2 != 0:
            b = np.append(b, 0)

        pairs = b.reshape(-1, 2)
        I = 1 - 2 * pairs[:, 0]
        Q = 1 - 2 * pairs[:, 1]

        return (I + 1j * Q) / np.sqrt(2)

    # =====================================================================
    #                       FULL PBCH ENCODING
    # =====================================================================
    def encode(self, info_bits: Sequence[int]) -> np.ndarray:
        """
        Kompletan PBCH encoding lanac:
            24 → 40 → 120 → 1920 → 960

        Parameters
        ----------
        info_bits : Sequence[int]
            Ulazni informacijski bitovi (24 bita).

        Returns
        -------
        numpy.ndarray
            QPSK simboli dužine 960.
        """
        info_bits = np.array(info_bits, dtype=np.uint8)
        assert len(info_bits) == 24, "Info bits must be 24 bits long!"

        # --- CRC16 (24 → 40)
        crc = self.crc16(info_bits)
        bits_40 = np.concatenate((info_bits, crc))

        # --- FEC rate 1/3 (40 → 120)
        bits_120 = self.fec_one_third(bits_40)

        # --- Rate matching (120 → 1920)
        bits_1920 = self.rate_match(bits_120, E=1920)

        # --- QPSK mapping (1920 → 960)
        symbols_960 = self.qpsk(bits_1920)

        if self.verbose:
            print("PBCH Encoding Summary:")
            print(f" Input bits        : {len(info_bits)}")
            print(f" After CRC16       : {len(bits_40)}")
            print(f" FEC output        : {len(bits_120)}")
            print(f" Rate matched bits : {len(bits_1920)}")
            print(f" QPSK symbols      : {len(symbols_960)}")

        return symbols_960


# ========================== Self-test ==========================
if __name__ == "__main__":
    info = np.random.randint(0, 2, 24)
    enc = PBCHEncoder(verbose=True)
    syms = enc.encode(info)
    print("Encoding OK, output length:", len(syms))
