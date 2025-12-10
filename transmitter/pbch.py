# ---------------- pbch.py ----------------
"""
PBCHEncoder: Python implementacija PBCH enkodiranja:
  CRC24A -> Convolutional fallback (TBC placeholder) -> Simplified rate matching -> QPSK mapping

Dependencies: numpy, matplotlib (za vizualizaciju)
"""

from typing import Sequence, Optional
import numpy as np
import math
import matplotlib.pyplot as plt

class PBCHEncoder:
    def __init__(self, target_bits: int = 384, enable_turbo: bool = False, verbose: bool = True):
        """
        Inicijalizacija PBCH enkodera.

        Parametri
        ----------
        target_bits : int
            Broj bitova nakon rate matchinga (E)
        enable_turbo : bool
            Ako je True, pokušava koristiti Turbo (nije implementirano, koristi fallback)
        verbose : bool
            Ako je True, ispisuje korake u terminalu
        """
        self.target_bits = int(target_bits)
        self.enable_turbo = bool(enable_turbo)
        self.verbose = bool(verbose)
        self._crc24a_poly = 0x1864CFB  # CRC24A polinom

    # ---------------- CRC24A ----------------
    def generate_crc24a(self, bits: Sequence[int]) -> np.ndarray:
        """
        Generiše CRC24A remainder za ulazni bit niz.

        Povratna vrijednost
        -----------------
        numpy.ndarray: CRC24A remainder (24 bita)

        Primjer
        -------
        >>> enc = PBCHEncoder()
        >>> crc = enc.generate_crc24a([1,0,1,1])
        >>> crc.shape
        (24,)
        """
        b = np.array(bits, dtype=np.uint8).flatten()
        reg = 0
        for bit in b:
            bit = int(bit) & 1
            top = ((reg >> 23) & 1)
            feedback = top ^ bit
            reg = ((reg << 1) & 0xFFFFFF)
            if feedback:
                reg ^= (self._crc24a_poly & 0xFFFFFF)
        rem = reg & 0xFFFFFF
        return np.array([(rem >> (23 - i)) & 1 for i in range(24)], dtype=np.uint8)

    # ---------------- Convolutional fallback ----------------
    def _conv_encode_fallback(self, bits: Sequence[int]) -> np.ndarray:
        """Fallback convolutional encoder (rate 1/2, K=7)."""
        b = np.array(bits, dtype=np.uint8).flatten()
        g0 = np.array([int(x) for x in format(int('171',8),'07b')], dtype=np.uint8)
        g1 = np.array([int(x) for x in format(int('133',8),'07b')], dtype=np.uint8)
        mem = np.zeros(6, dtype=np.uint8)
        out = []
        for bit in b:
            shiftreg = np.concatenate(([bit & 1], mem))
            out.append(int(np.sum(shiftreg * g0) % 2))
            out.append(int(np.sum(shiftreg * g1) % 2))
            mem = shiftreg[:-1]
        for _ in range(6):  # tail bits
            shiftreg = np.concatenate(([0], mem))
            out.append(int(np.sum(shiftreg * g0) % 2))
            out.append(int(np.sum(shiftreg * g1) % 2))
            mem = shiftreg[:-1]
        return np.array(out, dtype=np.uint8)

    def turbo_encode(self, bits: Sequence[int]) -> np.ndarray:
        """Placeholder za Turbo enkodiranje. Trenutno koristi convolutional fallback."""
        if self.enable_turbo and self.verbose:
            print("Turbo requested but not implemented, using convolutional fallback.")
        return self._conv_encode_fallback(bits)

    # ---------------- Rate matching ----------------
    def rate_match(self, coded_bits: Sequence[int], E: Optional[int] = None) -> np.ndarray:
        """Jednostavni rate matching: skraćuje ili ponavlja bitove da se dostigne E."""
        if E is None:
            E = self.target_bits
        cb = np.array(coded_bits, dtype=np.uint8).flatten()
        N = len(cb)
        if N == E:
            return cb.copy()
        elif N > E:
            idx = np.linspace(0, N-1, num=E, endpoint=True).round().astype(int)
            return cb[idx]
        else:
            reps = math.ceil(E / N)
            big = np.tile(cb, reps)
            return big[:E]

    # ---------------- QPSK mapping ----------------
    def qpsk_map(self, bits: Sequence[int]) -> np.ndarray:
        """Mapira binarne bitove na QPSK simbole (Gray mapping)."""
        b = np.array(bits, dtype=np.uint8).flatten()
        if len(b) % 2 != 0:
            b = np.concatenate((b, [0]))
        pairs = b.reshape(-1,2)
        I = 1.0 - 2.0 * pairs[:,0]
        Q = 1.0 - 2.0 * pairs[:,1]
        return (I + 1j * Q) / np.sqrt(2.0)

    # ---------------- Full PBCH lanac ----------------
    def encode(self, info_bits: Sequence[int]) -> np.ndarray:
        """Kompletan PBCH encoding lanac: CRC → TBC → Rate Matching → QPSK"""
        b = np.array(info_bits, dtype=np.uint8).flatten()
        crc = self.generate_crc24a(b)
        bits_crc = np.concatenate((b, crc))
        coded = self.turbo_encode(bits_crc)
        rm = self.rate_match(coded, E=self.target_bits)
        symbols = self.qpsk_map(rm)
        if self.verbose:
            print(f"CRC appended: {len(bits_crc)} bits")
            print(f"Coded bits: {len(coded)} bits")
            print(f"Rate matched to {len(rm)} bits")
            print(f"Mapped to {len(symbols)} QPSK symbols")
        return symbols

# ---------------- Vizualizacija ----------------
def plot_qpsk_constellation(symbols):
    """Prikazuje QPSK konstelaciju."""
    plt.figure(figsize=(6,6))
    plt.scatter(symbols.real, symbols.imag, color='blue')
    plt.title('PBCH QPSK Constellation')
    plt.xlabel('I (In-phase)')
    plt.ylabel('Q (Quadrature)')
    plt.grid(True)
    plt.axis('equal')
    plt.show(block=True)

def plot_eye_diagram(symbols, samples_per_symbol=4):
    """Prikazuje Eye dijagram (I i Q na istom grafiku)."""
    sig = symbols.flatten()
    L = samples_per_symbol
    num_traces = len(sig) // L
    plt.figure(figsize=(8,4))
    for i in range(num_traces):
        plt.plot(sig[i*L:(i+1)*L].real, color='blue', alpha=0.5)
        plt.plot(sig[i*L:(i+1)*L].imag, color='red', alpha=0.5)
    plt.title('Eye Diagram (I i Q components)')
    plt.xlabel('Sample index within symbol period')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(['I component','Q component'])
    plt.show(block=True)

# ---------------- Glavni test (automatski vizualizira) ----------------
if __name__ == "__main__":
    # Nasumični PBCH info bits (npr. 24)
    info_bits = np.random.randint(0,2,24)
    encoder = PBCHEncoder(target_bits=384, verbose=True)

    # 1. Encode PBCH
    symbols = encoder.encode(info_bits)

    # 2. Vizualizacije
    plot_qpsk_constellation(symbols)
    plot_eye_diagram(symbols, samples_per_symbol=4)

    # ---------------- Ispis detalja koraka ----------------
    print("\n--- PBCH Encoding Details ---")
    print(f"Original info bits ({len(info_bits)}): {info_bits}")

    # CRC
    crc = encoder.generate_crc24a(info_bits)
    bits_crc = np.concatenate((info_bits, crc))
    print(f"After CRC appended ({len(bits_crc)}): {bits_crc}")

    # TBC / fallback
    coded = encoder.turbo_encode(bits_crc)
    print(f"After convolutional fallback ({len(coded)}): {coded}")

    # Rate Matching
    rm_bits = encoder.rate_match(coded, E=encoder.target_bits)
    print(f"After rate matching ({len(rm_bits)}): {rm_bits}")

    # QPSK mapping
    print(f"Mapped to {len(symbols)} QPSK symbols (first 10): {symbols[:10]}")

    print("\nPBCH encoding complete!")
