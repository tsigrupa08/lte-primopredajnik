"""
LTE Transmit Chain Module

Implements LTE downlink transmit chain:
- PSS generation and mapping
- MIB → PBCH QPSK encoding and mapping
- OFDM modulation
"""

import numpy as np
from transmitter.pss import PSSGenerator
from transmitter.resource_grid import create_resource_grid, map_pss_to_grid, map_pbch_to_grid
from transmitter.ofdm import OFDMModulator
from transmitter.pbch import PBCHEncoder


class LTETxChain:
    """
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
        self.grid = create_resource_grid(
            ndlrb=self.ndlrb,
            num_subframes=self.num_subframes,
            normal_cp=self.normal_cp,
        )

    def generate_waveform(self, mib_bits=None):
        """
        Generate LTE OFDM waveform.

        Parameters
        ----------
        mib_bits : np.ndarray of int, optional
            Binary MIB sequence (1920 bits) used for PBCH.

        Returns
        -------
        waveform : np.ndarray
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

        return waveform, fs
