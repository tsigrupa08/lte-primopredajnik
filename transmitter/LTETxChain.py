"""
LTE Transmit Chain Module (TX)

<<<<<<< HEAD
Implements LTE downlink transmit chain:
- PSS generation and mapping
- MIB → PBCH QPSK encoding and mapping
- OFDM modulation
=======
Ovaj modul implementira pojednostavljeni LTE downlink predajni lanac:
- Generisanje PSS sekvence i mapiranje u resource grid
- MIB (24 bita) -> PBCH enkodiranje -> QPSK simboli
- Mapiranje PBCH simbola u resource grid
- OFDM modulacija (IFFT + CP)

Napomena
--------
Ovaj projekat koristi pojednostavljeni pristup PBCH mapiranju:
- PBCH enkoder proizvodi 960 QPSK simbola (1920 bita -> 960 QPSK).
- Ovi 960 simbola se mapiraju kao 4 bloka po 240 simbola kroz 4 subfrejma.
  (240 simbola po subfrejmu je tipično za PBCH regiju nakon izostavljanja nekih RE,
   ali ovdje mapiramo sekvencijalno bez CRS maske osim ako je ne proslijediš.)
>>>>>>> d451b4f (Fix LTETxChain TX pipeline)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
<<<<<<< HEAD
from transmitter.pss import PSSGenerator
from transmitter.resource_grid import create_resource_grid, map_pss_to_grid, map_pbch_to_grid
from transmitter.ofdm import OFDMModulator
=======

from transmitter.pss import PSSGenerator
from transmitter.resource_grid import create_resource_grid, map_pss_to_grid, map_pbch_to_grid
>>>>>>> d451b4f (Fix LTETxChain TX pipeline)
from transmitter.pbch import PBCHEncoder
from transmitter.ofdm import OFDMModulator


class LTETxChain:
    """
<<<<<<< HEAD
    LTE downlink transmit chain.
    """

    def __init__(self, n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True):
        self.n_id_2 = n_id_2
        self.ndlrb = ndlrb
        self.num_subframes = num_subframes
        self.normal_cp = normal_cp

        # Kreiranje praznog grid-a
        self.grid = None
        self._reset_grid()

    def _reset_grid(self):
        """Create an empty LTE resource grid."""
=======
    LTE downlink predajni lanac (TX).

    Parametri
    ---------
    n_id_2 : int, optional
        PSS identitet N_ID_2 u {0, 1, 2}. Default je 0.
    ndlrb : int, optional
        Broj downlink RB-ova. Za 1.4 MHz LTE, ndlrb=6. Default je 6.
    num_subframes : int, optional
        Broj subfrejmova koji se alociraju u resource gridu. Default je 1.
        VAŽNO: Ako želiš mapirati kompletan PBCH od 960 simbola, postavi num_subframes >= 4.
    normal_cp : bool, optional
        True za normalni CP (14 simbola/subfrejm), False za prošireni CP (12 simbola/subfrejm).

    Atributi
    --------
    grid : np.ndarray
        Kompleksni resource grid, shape (12*ndlrb, ukupno_ofdm_simbola).
    """

    def __init__(
        self,
        n_id_2: int = 0,
        ndlrb: int = 6,
        num_subframes: int = 1,
        normal_cp: bool = True,
    ) -> None:
        self.n_id_2 = int(n_id_2)
        self.ndlrb = int(ndlrb)
        self.num_subframes = int(num_subframes)
        self.normal_cp = bool(normal_cp)

        self.grid: np.ndarray
        self._reset_grid()

    # ---------------------------------------------------------------------
    # Interne pomoćne funkcije
    # ---------------------------------------------------------------------
    @property
    def symbols_per_subframe(self) -> int:
        """Broj OFDM simbola po subfrejmu (14 za normal CP, 12 za prošireni CP)."""
        return 14 if self.normal_cp else 12

    def _reset_grid(self) -> None:
        """Kreira prazan LTE resource grid za trenutnu konfiguraciju."""
>>>>>>> d451b4f (Fix LTETxChain TX pipeline)
        self.grid = create_resource_grid(
            ndlrb=self.ndlrb,
            num_subframes=self.num_subframes,
            normal_cp=self.normal_cp,
        )

    def _pss_symbol_index(self) -> int:
        """
<<<<<<< HEAD
        Generate LTE OFDM waveform.

        Parameters
        ----------
        mib_bits : np.ndarray of int, optional
            Binary MIB sequence (1920 bits) used for PBCH.
=======
        Indeks OFDM simbola za PSS unutar subfrejma 0.

        Normal CP: zadnji simbol slota 0 -> l = 6 (slot0: 0..6)
        Prošireni CP: zadnji simbol slota 0 -> l = 5 (slot0: 0..5)
        """
        return 6 if self.normal_cp else 5

    def _pbch_symbol_indices_for_subframe(self, subframe_index: int) -> list[int]:
        """
        Indeksi OFDM simbola (kolone) za PBCH u datom subfrejmu.
>>>>>>> d451b4f (Fix LTETxChain TX pipeline)

        Za subfrejm 0 (i analogno za ostale ako mapiramo blokove kroz subfrejmove):
        Normal CP: PBCH u prva 4 simbola slota 1 -> l = 7,8,9,10
        Prošireni CP: slot ima 6 simbola, slot 1 počinje na l=6 -> l = 6,7,8,9
        """
        base = subframe_index * self.symbols_per_subframe
        if self.normal_cp:
            return [base + 7, base + 8, base + 9, base + 10]
        return [base + 6, base + 7, base + 8, base + 9]

    # ---------------------------------------------------------------------
    # Javni API
    # ---------------------------------------------------------------------
    def generate_waveform(
        self,
        mib_bits: Optional[Sequence[int]] = None,
        reserved_re_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Generiše LTE OFDM talasni oblik (waveform).

        Koraci
        ------
        1) Reset grida
        2) Generiši PSS i mapiraj u subfrejm 0
        3) Ako je mib_bits zadano:
           - PBCH enkodiranje (24 -> ... -> 960 QPSK simbola)
           - Mapiranje 960 simbola kao 4 bloka po 240 kroz 4 subfrejma
        4) OFDM modulacija

        Parametri
        ---------
        mib_bits : Sequence[int] ili None, optional
            MIB informacijski bitovi dužine 24 (0/1). Ako je None, PBCH se ne mapira.
        reserved_re_mask : np.ndarray ili None, optional
            Opciona bool maska rezervisanih resource elemenata (isti shape kao grid).
            Ako je zadana, PBCH mapiranje preskače pozicije gdje je maska True.

        Povratna vrijednost
        -------------------
        waveform : np.ndarray
<<<<<<< HEAD
            Time-domain OFDM signal.
        fs : float
            Sampling frequency.
        """
        # Reset grid pri svakom pozivu
        self._reset_grid()

        # PSS generacija (OOP)
        pss = PSSGenerator.generate(self.n_id_2)
        pss_symbol = 6 if self.normal_cp else 5

        map_pss_to_grid(
            self.grid,
            pss,
            symbol_index=pss_symbol,
            ndlrb=self.ndlrb,
        )

        # PBCH mapiranje
        if mib_bits is not None:
            if len(mib_bits) != 1920:
                raise ValueError("PBCH expects exactly 1920 MIB bits.")

            pbch_encoder = PBCHEncoder(verbose=False)
            pbch_symbols = pbch_encoder.encode(mib_bits)

            # PBCH simboli idu na ispravne indekse
            pbch_symbol_indices = [6, 7, 8, 9] if self.normal_cp else [5, 6, 7, 8]

            map_pbch_to_grid(
                self.grid,
                pbch_symbols,
                pbch_symbol_indices=pbch_symbol_indices,
                ndlrb=self.ndlrb,
            )

        # OFDM modulacija
        ofdm = OFDMModulator(self.grid)
        waveform, fs = ofdm.modulate()

        # Provjera shape-a
        if waveform.shape[0] != ofdm.output_length:
            print("[Warning] Output waveform length ne odgovara očekivanom output_length.")
=======
            Kompleksni vremenski OFDM signal (sa CP).
        fs : float
            Frekvencija uzorkovanja u Hz.

        Izuzeci
        -------
        ValueError
            Ako n_id_2 nije u {0,1,2}.
            Ako mib_bits nema dužinu 24.
            Ako je mib_bits zadano, a num_subframes < 4 (ne može se smjestiti 960 simbola kao 4×240).
            Ako reserved_re_mask nema isti shape kao grid.
        """
        # Reset grid pri svakom pozivu (čisto generisanje)
        self._reset_grid()

        # Validacija N_ID_2 odmah (jasna poruka greške)
        if self.n_id_2 not in (0, 1, 2):
            raise ValueError("n_id_2 mora biti 0, 1 ili 2.")

        if reserved_re_mask is not None and reserved_re_mask.shape != self.grid.shape:
            raise ValueError("reserved_re_mask mora imati isti shape kao grid.")

        # 1) PSS generisanje + mapiranje
        pss = PSSGenerator.generate(self.n_id_2)
        map_pss_to_grid(
            self.grid,
            pss,
            symbol_index=self._pss_symbol_index(),
            ndlrb=self.ndlrb,
        )

        # 2) PBCH enkodiranje + mapiranje (opciono)
        if mib_bits is not None:
            bits = np.asarray(mib_bits, dtype=np.uint8).flatten()
            if bits.size != 24:
                raise ValueError("MIB mora imati tačno 24 bita (0/1).")

            # Enkoder treba dati 960 QPSK simbola (prema tvojoj projektnoj šemi)
            encoder = PBCHEncoder(verbose=False)
            pbch_symbols = encoder.encode(bits)

            pbch_symbols = np.asarray(pbch_symbols).flatten()
            if pbch_symbols.size != 960:
                raise ValueError(
                    f"PBCHEncoder.encode() treba vratiti 960 QPSK simbola, "
                    f"ali je vratio {pbch_symbols.size}."
                )

            # Trebaju 4 subfrejma da se smjesti 4 bloka po 240 simbola
            if self.num_subframes < 4:
                raise ValueError(
                    "Za mapiranje 960 PBCH simbola (4×240) postavi num_subframes >= 4."
                )

            # Mapiranje 4 bloka (240 po subfrejmu)
            for sf in range(4):
                chunk = pbch_symbols[sf * 240 : (sf + 1) * 240]
                l_idx = self._pbch_symbol_indices_for_subframe(sf)

                map_pbch_to_grid(
                    self.grid,
                    chunk,
                    pbch_symbol_indices=l_idx,
                    ndlrb=self.ndlrb,
                    reserved_re_mask=reserved_re_mask,
                )

        # 3) OFDM modulacija
        ofdm = OFDMModulator(self.grid)
        waveform, fs = ofdm.modulate()
>>>>>>> d451b4f (Fix LTETxChain TX pipeline)

        return waveform, fs


# ---------------------------------------------------------------------
# Primjer korištenja
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Primjer 1: Samo PSS (1 subfrejm je dovoljan)
    tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
    w, fs = tx.generate_waveform()
    print("Waveform (samo PSS):", w.shape, "fs =", fs)

    # Primjer 2: PSS + PBCH (treba >=4 subfrejma za 960 PBCH simbola)
    tx2 = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=4, normal_cp=True)
    mib = np.random.randint(0, 2, 24)
    w2, fs2 = tx2.generate_waveform(mib_bits=mib)
    print("Waveform (PSS+PBCH):", w2.shape, "fs =", fs2)
