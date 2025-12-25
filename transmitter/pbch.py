"""
PBCHEncoder
===========

Pojednostavljeni PBCH enkoder prema projektnoj specifikaciji:

    24 → 40 → 120 → 1920 → 960

Gdje je:
- 24  = informacijski bitovi
- 40  = informacijski + CRC16
- 120 = FEC kodirani bitovi (rate 1/3, CONVOLUTIONAL)
- 1920 = nakon rate-matchinga
- 960 = broj QPSK simbola

FEC dio je sada kompatibilan sa Viterbi dekoderom u prijemniku.
"""

from typing import Sequence
import numpy as np
import math

#pravi konvolucijski FEC
from transmitter.convolutional import ConvolutionalEncoder


class PBCHEncoder:
    """
    PBCHEncoder: kompletan enkoder za PBCH zadat iz projektne dokumentacije.

    Parameters
    ----------
    verbose : bool
        Ako je True, ispisuje informacije o svakom koraku enkodiranja.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        # instanca konvolucijskog enkodera (rate 1/3)
        self.fec = ConvolutionalEncoder(
            constraint_len=7,
            generators_octal=(0o133, 0o171, 0o164),
            tail_biting=False,
        )

    # =====================================================================
    #                           CRC16
    # =====================================================================
    def crc16(self, bits: Sequence[int]) -> np.ndarray:
        """
        CRC16 (polinom 0x1021, init 0xFFFF)
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
    #                       RATE MATCHING
    # =====================================================================
    def rate_match(self, bits: Sequence[int], E: int = 1920) -> np.ndarray:
        """
        Rate matching:
        - repeat ako je ulaz kraći
        - subsample ako je duži
        """
        b = np.asarray(bits, dtype=np.uint8)
        N = b.size

        if N == E:
            return b.copy()

        if N < E:
            reps = math.ceil(E / N)
            return np.tile(b, reps)[:E]

        idx = np.linspace(0, N - 1, num=E).round().astype(int)
        return b[idx]

    # =====================================================================
    #                       QPSK MAPPING
    # =====================================================================
    def qpsk(self, bits: Sequence[int]) -> np.ndarray:
        """
        Gray QPSK mapiranje
        """
        b = np.asarray(bits, dtype=np.uint8)

        if b.size % 2 != 0:
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
        """
        info_bits = np.asarray(info_bits, dtype=np.uint8)
        assert info_bits.size == 24, "Info bits must be 24 bits long!"

        # 24 → 40 (CRC)
        bits_40 = np.concatenate((info_bits, self.crc16(info_bits)))

        # 40 → 120 (PRAVI konvolucijski FEC)
        bits_120 = self.fec.encode(bits_40)

        # 120 → 1920 (rate matching)
        bits_1920 = self.rate_match(bits_120, E=1920)

        # 1920 → 960 (QPSK)
        symbols_960 = self.qpsk(bits_1920)

        if self.verbose:
            print("PBCH Encoding Summary:")
            print(f" Input bits        : {info_bits.size}")
            print(f" After CRC16       : {bits_40.size}")
            print(f" FEC output        : {bits_120.size}")
            print(f" Rate matched bits : {bits_1920.size}")
            print(f" QPSK symbols      : {symbols_960.size}")

        return symbols_960


# ========================== Self-test ==========================
if __name__ == "__main__":
    info = np.random.randint(0, 2, 24, dtype=np.uint8)
    enc = PBCHEncoder(verbose=True)
    syms = enc.encode(info)

    print("Encoding OK, output length:", len(syms))  # mora biti 960
