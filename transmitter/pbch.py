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
    def __init__(self, verbose: bool = True, pci: int = 0, enable_scrambling: bool = True):
        self.verbose = verbose
        self.pci = int(pci)
        self.enable_scrambling = bool(enable_scrambling)

        self.fec = ConvolutionalEncoder(
            constraint_len=7,
            generators_octal=(0o133, 0o171, 0o165),
            tail_biting=True,
        )

   
    # =====================================================================
    #                           CRC16
    # =====================================================================
    def crc16(self, bits: Sequence[int]) -> np.ndarray:
        poly = 0x1021
        reg = 0xFFFF
        for b in bits:
            b = int(b) & 1
            xor = ((reg >> 15) & 1) ^ b
            reg = ((reg << 1) & 0xFFFF)
            if xor:
                reg ^= poly
        return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


    # =====================================================================
    #                       RATE MATCHING (LTE PBCH)
    # =====================================================================

    # Permutation pattern (table 5.1.4-2)
    _PERM_PATTERN = np.array(
        [1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
        0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30],
        dtype=int
    )

    def _subblock_interleave_120(self, bits_120: Sequence[int]) -> np.ndarray:
        """
        LTE PBCH sub-block interleaver for 120 coded bits (rate 1/3).
        - Split into 3 streams d0,d1,d2 (each 40 bits)
        - For each stream: 32-column interleaver with dummy bits at the start,
        permute columns with _PERM_PATTERN, then read out column-wise.
        Returns 120 interleaved bits.
        """
        b = np.asarray(bits_120, dtype=np.uint8).reshape(-1)
        assert b.size == 120, "PBCH rate matching expects 120 coded bits."

        # Split encoded bits into 3 streams (length 40 each)
        streams = [b[0::3], b[1::3], b[2::3]]

        C = 32  # NumColumns
        out_streams = []

        for s in streams:
            Nin = s.size  # 40
            R = int(np.ceil(Nin / C))     # NumRows = 2
            Kpi = C * R                   # TotalNumberOfTableEntries = 64
            Nd = Kpi - Nin                # NumDummyBits = 24

            # Dummy elements inserted at the START (use -1 sentinel)
            linear = np.concatenate(
                (np.full(Nd, -1, dtype=np.int16), s.astype(np.int16))
            )

            # Match Matlab: reshape(linear, NumColumns, NumRows)'  with column-major (Fortran) order
            mat = linear.reshape((C, R), order="F").T   # shape (R, C) = (2, 32)

            # Permute columns
            mat = mat[:, self._PERM_PATTERN]

            # Match Matlab: Outputindex = mat(:) (column-wise)
            out = mat.flatten(order="F")

            # Prune dummy elements
            out = out[out != -1].astype(np.uint8)  # length 40
            out_streams.append(out)

        interleaved = np.concatenate(out_streams)  # 120
        assert interleaved.size == 120
        return interleaved

    def rate_match(self, bits_120: Sequence[int], E: int = 1920) -> np.ndarray:
        """
        PBCH rate matching = interleaving + repetition.

        Normal CP:  120 interleaved bits repeated 16 times -> 1920
        Extended CP: repeat 14 times then append first 48 bits -> 1728
        """
        interleaved = self._subblock_interleave_120(bits_120)

        if E == 1920:
            return np.tile(interleaved, 16).astype(np.uint8)
        elif E == 1728:
            return np.concatenate((np.tile(interleaved, 14), interleaved[:48])).astype(np.uint8)
        else:
            raise ValueError("PBCH rate matching supports only E=1920 (normal CP) or E=1728 (extended CP).")

    def gold_sequence_pbch(self, c_init: int, length: int) -> np.ndarray:
        """
        Gold scrambler za PBCH:
        - x1: init [1, 0, 0, ..., 0]
        - x2: init iz c_init (PCI), 31 bita
        - 1600 warm-up iteracija
        - c[n] = x1[n] XOR x2[n]
        """
        c_init = int(c_init)
        assert 0 <= c_init <= 503, "PCI (c_init) mora biti u [0, 503] za LTE PBCH."

        # x1 init
        x1 = np.zeros(31, dtype=np.uint8)
        x1[0] = 1

        # x2 init: bit i (LSB-first) -> x2[i]
        x2 = np.array([(c_init >> i) & 1 for i in range(31)], dtype=np.uint8)

        def step(reg: np.ndarray, taps: Sequence[int]) -> np.ndarray:
            # reg[0]=x[n], reg[3]=x[n+3] ...
            new_bit = np.uint8(0)
            for t in taps:
                new_bit ^= reg[t]
            reg = np.roll(reg, -1)
            reg[-1] = new_bit
            return reg

        # Polinomi (standardni LTE Gold):
        # x1(n+31) = x1(n+3) + x1(n)
        # x2(n+31) = x2(n+3) + x2(n+2) + x2(n+1) + x2(n)
        for _ in range(1600):
            x1 = step(x1, (0, 3))
            x2 = step(x2, (0, 1, 2, 3))

        out = np.zeros(length, dtype=np.uint8)
        for n in range(length):
            out[n] = x1[0] ^ x2[0]
            x1 = step(x1, (0, 3))
            x2 = step(x2, (0, 1, 2, 3))

        return out

    # =====================================================================
    #                       QPSK MAPPING
    # =====================================================================
    def qpsk(self, bits):
        """
        Gray QPSK mapiranje: 0->+1, 1->-1 (po I i Q grani)
        """
        b = np.asarray(bits, dtype=np.int8).ravel()

        if b.size % 2 != 0:
            b = np.append(b, 0).astype(np.int8)

        pairs = b.reshape(-1, 2)

        I = 1.0 - 2.0 * pairs[:, 0].astype(np.float64)
        Q = 1.0 - 2.0 * pairs[:, 1].astype(np.float64)

        return ((I + 1j * Q) / np.sqrt(2.0)).astype(np.complex64)


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

        # Scrambling (opcijski)
        if self.enable_scrambling:
            c = self.gold_sequence_pbch(self.pci, bits_1920.size)
            bits_1920 = (bits_1920 ^ c).astype(np.uint8)

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
