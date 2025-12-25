"""
crc_checker.py
==============

CRC provjera za PBCH RX lanac.

CRC implementacija je 1:1 preuzeta iz PBCH TX enkodera.
"""

from __future__ import annotations
import numpy as np


class CRCChecker:
    """
    CRC checker for PBCH.

    Parameters
    ----------
    poly : int
        CRC polinom (npr. 0x1021).
    init : int
        Početna vrijednost CRC registra (npr. 0xFFFF).
    """

    def __init__(self, poly: int = 0x1021, init: int = 0xFFFF):
        self.poly = poly
        self.init = init

    # ------------------------------------------------------------------
    # CRC16 (1:1 PREUZETO IZ PBCHEncoder)
    # ------------------------------------------------------------------
    def _crc16(self, bits: np.ndarray) -> np.ndarray:
        """
        Izračunava CRC16 remainder nad ulaznim bitovima.

        Parameters
        ----------
        bits : np.ndarray
            Ulazni bitovi (0/1).

        Returns
        -------
        np.ndarray
            CRC remainder (16 bita).
        """
        reg = self.init

        for b in bits:
            reg ^= (int(b) & 1) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ self.poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF

        return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)

    # ------------------------------------------------------------------
    # CHECK
    # ------------------------------------------------------------------
    def check(self, bits_with_crc: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Provjerava CRC nad PBCH payload-om.

        Parameters
        ----------
        bits_with_crc : np.ndarray
            Bitovi koji sadrže payload + CRC (npr. 40 bita).

        Returns
        -------
        payload_bits : np.ndarray
            Informacijski bitovi (bez CRC-a).
        ok : bool
            True ako CRC prolazi, False inače.
        """
        bits = np.asarray(bits_with_crc, dtype=np.uint8)

        if bits.size < 16:
            raise ValueError("Ulaz mora sadržavati barem 16 CRC bitova.")

        payload = bits[:-16]
        crc_rx = bits[-16:]

        crc_calc = self._crc16(payload)

        ok = np.array_equal(crc_calc, crc_rx)
        return payload, ok
