"""
de_rate_matching.py

PBCH de-rate matching za tvoj TX:

TX:
  120 (interleaved) -> repetition -> E (1920 normal CP) / (1728 extended CP)

RX:
  E -> 120 (interleaved)  [samo sklapanje ponavljanja]

NAPOMENA:
- OVO vraća 120 *interleaved* kodiranih bitova.
  Nakon ovoga ti još treba PBCH de-interleaver (inverz od TX sub-block interleavera),
  pa tek onda Viterbi dekodiranje (rate 1/3, tail-biting).
- De-rate matching radi smisleno tek nakon descrambling-a (ako je TX imao scrambling).
"""

from __future__ import annotations
import numpy as np


class DeRateMatcherPBCH:
    def __init__(self, n_coded: int = 120) -> None:
        self.n_coded = int(n_coded)
        if self.n_coded <= 0:
            raise ValueError("n_coded mora biti > 0.")

    def derate_match(self, bits_rx, *, return_soft: bool = False) -> np.ndarray:
        """
        Sklapa E primljenih bitova nazad na n_coded=120.

        bits_rx:
          - očekuje hard bitove 0/1 (np.uint8/int) nakon QPSK demapiranja
          - može biti i float (npr. 0..1), tada se tretira kao "mekani" dokaz,
            ali za tvoj projekat je dovoljno da bude 0/1.

        return_soft:
          - False: vraća hard 0/1 (majority vote)
          - True : vraća soft vrijednosti (suma dokaza po bitu), korisno ako kasnije želiš soft-Viterbi
        """
        x = np.asarray(bits_rx).ravel()
        if x.size < self.n_coded:
            raise ValueError(f"Ulaz prekratak: E={x.size}, očekujem bar {self.n_coded} bitova.")

        E = int(x.size)

        # Mapiranje indeksa ponavljanja: i -> i % 120
        idx = np.arange(E, dtype=np.int64) % self.n_coded

        # Broj ponavljanja po poziciji (važno za E=1728 gdje prvih 48 ima +1 ponavljanje)
        counts = np.bincount(idx, minlength=self.n_coded).astype(np.float64)

        # Pretvori hard bitove 0/1 u "dokaz" (+1 za 0, -1 za 1), pa saberi.
        # (Za normal CP je svejedno sum/avg, ali za extended CP sum čuva veću pouzdanost prvih 48.)
        if np.issubdtype(x.dtype, np.floating):
            # Ako dođe float, pretpostavi da je u [0,1] (probabilistički ili “soft bit”),
            # mapiraj oko 0.5: <0.5 -> +, >0.5 -> -
            evidence = (0.5 - x.astype(np.float64)) * 2.0
        else:
            b = (x.astype(np.int8) & 1)
            evidence = 1.0 - 2.0 * b  # 0->+1, 1->-1

        summed = np.bincount(idx, weights=evidence.astype(np.float64), minlength=self.n_coded)

        if return_soft:
            # Soft vrijednosti (što je veće pozitivno -> više naginje bitu 0)
            # Po želji možeš i normalizovati: summed / counts
            return summed

        # Hard odluka: ako je suma negativna -> više "1" nego "0"
        out_hard = (summed < 0).astype(np.uint8)
        return out_hard


# Brzi sanity test
if __name__ == "__main__":
    drm = DeRateMatcherPBCH(n_coded=120)

    # Simuliraj idealno TX ponavljanje (normal CP): 120 -> 1920
    bits120 = np.random.randint(0, 2, 120, dtype=np.uint8)
    bits1920 = np.tile(bits120, 16)

    rec120 = drm.derate_match(bits1920)
    print("OK:", np.all(rec120 == bits120))
