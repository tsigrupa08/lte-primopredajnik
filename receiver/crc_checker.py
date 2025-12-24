"""
crc_checker.py
==============

CRC provjera za PBCH RX lanac.

CRC implementacija je 1:1 preuzeta iz PBCH TX enkodera
i koristi se na prijemnoj strani za provjeru ispravnosti
dekodiranih PBCH bitova.
"""

from __future__ import annotations
import numpy as np


class CRCChecker:
    """
    CRC checker za PBCH (RX strana).

    Parameters
    ----------
    poly : int, optional
        CRC polinom (default: 0x1021, CRC-16-CCITT).
    init : int, optional
        Početna vrijednost CRC registra (default: 0xFFFF).

    Examples
    --------
    >>> import numpy as np
    >>> from rx.crc_checker import CRCChecker
    >>>
    >>> payload = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    >>> crc = CRCChecker()
    >>>
    >>> crc_bits = crc._crc16(payload)
    >>> bits_with_crc = np.concatenate([payload, crc_bits])
    >>>
    >>> payload_rx, ok = crc.check(bits_with_crc)
    >>> ok
    True
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
            Ulazni bitovi (0/1), shape (N,).

        Returns
        -------
        np.ndarray
            CRC remainder (16 bita), shape (16,).

        Notes
        -----
        - Implementacija je bit-po-bit.
        - Identična je CRC funkciji korištenoj u PBCH TX enkoderu.
        """
        bits = np.asarray(bits, dtype=np.uint8).flatten()

        reg = self.init

        for b in bits:
            reg ^= (int(b) & 1) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ self.poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF

        return np.array(
            [(reg >> (15 - i)) & 1 for i in range(16)],
            dtype=np.uint8
        )

    # ------------------------------------------------------------------
    # CRC CHECK
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

        Examples
        --------
        >>> import numpy as np
        >>> crc = CRCChecker()
        >>> bits = np.array([1, 0, 1, 0], dtype=np.uint8)
        >>> bits_crc = np.concatenate([bits, crc._crc16(bits)])
        >>> payload, ok = crc.check(bits_crc)
        >>> ok
        True
        """
        bits = np.asarray(bits_with_crc, dtype=np.uint8).flatten()

        if bits.size < 16:
            raise ValueError("Ulaz mora sadržavati barem 16 CRC bitova.")

        payload = bits[:-16]
        crc_rx = bits[-16:]

        crc_calc = self._crc16(payload)
        ok = np.array_equal(crc_calc, crc_rx)

        return payload, ok


# ------------------------------------------------------------------
# Jednostavan lokalni test (opcionalno)
# ------------------------------------------------------------------
if __name__ == "__main__":
    payload = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    crc = CRCChecker()

    bits_with_crc = np.concatenate([payload, crc._crc16(payload)])
    payload_rx, ok = crc.check(bits_with_crc)

    print("Payload RX :", payload_rx)
    print("CRC OK     :", ok)
