"""
LTE Transmit Chain Module

Ovaj modul implementira predajni lanac LTE signala (TX chain) koristeći:
- generisanje PSS sekvence,
- enkodiranje MIB → PBCH QPSK,
- mapiranje PSS i PBCH u resource grid,
- OFDM modulaciju.

"""

import numpy as np

from transmitter.pss import generate_pss_sequence
from transmitter.resource_grid import (
    create_resource_grid,
    map_pss_to_grid,
    map_pbch_to_grid,
)
from transmitter.ofdm import OFDMModulator
from transmitter.pbch import PBCHEncoder


class LTETxChain:
    """
    Implementacija LTE downlink predajnog lanca.

    Ova klasa izvodi standardne korake LTE transmisije:
    generisanje PSS signala, enkodiranje MIB → PBCH, mapiranje simbola u resource grid,
    te OFDM modulaciju.

    Parameters
    ----------
    nid2 : int, optional
        LTE fizički identitet celije (NID2), vrijednost 0–2.
        Default je 0.
    ndlrb : int, optional
        Broj downlink resurs blokova (N_DL_RB). Minimalna vrijednost je 6 (1.4 MHz).
        Default je 6.
    num_subframes : int, optional
        Broj subframe-ova koji se generišu u resource gridu. Default je 1.
    normal_cp : bool, optional
        Ako je True, koristi se normal cyclic prefix, u suprotnom extended CP. Default je True.

    Attributes
    ----------
    grid : np.ndarray
        Kreirani LTE resource grid u frekvencijskoj domeni.
    """

    def __init__(self, nid2=0, ndlrb=6, num_subframes=1, normal_cp=True):
        self.nid2 = nid2
        self.ndlrb = ndlrb
        self.num_subframes = num_subframes
        self.normal_cp = normal_cp

        self.grid = create_resource_grid(
            ndlrb=self.ndlrb,
            num_subframes=self.num_subframes,
            normal_cp=self.normal_cp,
        )

    def generate_waveform(self, mib_bits=None):
        """
        Generiše kompletan LTE OFDM signal sa PSS i opcionalnim PBCH.

        Proces uključuje:
        1. Generisanje PSS sekvence
        2. Mapiranje PSS-a u resource grid (simbol 6 u slotu 0)
        3. PBCH enkodiranje MIB bitova i mapiranje u grid (simboli 0-3 u slotu 0)
        4. OFDM modulaciju rezultirajućeg grida

        Parameters
        ----------
        mib_bits : np.ndarray or list of int, optional
            Binarni niz MIB bitova za PBCH. Ako nije None, kodira i mapira PBCH.

        Returns
        -------
        waveform : np.ndarray
            Kompleksni signal u vremenskoj domeni (OFDM izlaz).
        fs : float
            Sample rate korišten za OFDM modulaciju.
        """

        # 1) PSS generacija
        pss, _ = generate_pss_sequence(self.nid2)
        map_pss_to_grid(self.grid, pss, symbol_index=6, ndlrb=self.ndlrb)

        # 2) PBCH enkodiranje i mapiranje
        if mib_bits is not None:
            pbch_encoder = PBCHEncoder(target_bits=384, verbose=False)
            pbch_symbols = pbch_encoder.encode(mib_bits)
            # mapiranje PBCH u grid: simboli 0–3 u slotu 0 (LTE specifikacija)
            map_pbch_to_grid(
                self.grid,
                pbch_symbols,
                pbch_symbol_indices=[0, 1, 2, 3],
                ndlrb=self.ndlrb
            )

        # 3) OFDM modulacija
        modulator = OFDMModulator(self.grid)
        waveform, fs = modulator.modulate()

        return waveform, fs
