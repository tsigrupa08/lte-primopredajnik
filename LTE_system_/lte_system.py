"""
lte_system.py
=============

End-to-end LTE sistem koji povezuje kompletan predajni i prijemni lanac:

    TX (LTETxChain) → Channel (LTEChannel) → RX (LTERxChain)

Modul služi kao “glue layer” između TX, Channel i RX blokova i
namijenjen je demonstraciji rada LTE sistema u cjelini
(kroz demo skripte i GUI).

Glavna klasa
------------
LTESystem

    LTESystem(tx, ch, rx)

Javni API
---------
run(mib_bits) -> dict

Metoda `run` izvršava kompletnu end-to-end simulaciju:
    - generisanje LTE signala (TX),
    - prolazak kroz kanal (CFO + AWGN),
    - prijem, sinkronizaciju i dekodiranje (RX).

Vraća rječnik prilagođen vizualizaciji i GUI aplikacijama.

Napomena
--------
Sva obrada je delegirana na postojeće TX, Channel i RX module.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence

import numpy as np


# ======================================================================
# Results container
# ======================================================================
@dataclass
class LTESystemResults:
    """
    Struktura rezultata end-to-end LTE simulacije.
    """

    # --- core I/O ---
    mib_bits_tx: np.ndarray
    mib_bits_rx: Optional[np.ndarray]

    tx_waveform: np.ndarray
    rx_waveform: np.ndarray
    fs_hz: float

    # --- quality metrics ---
    crc_ok: bool
    bit_errors: Optional[int]
    ber: Optional[float]

    # --- sync / CFO ---
    detected_nid: Optional[int]
    tau_hat: Optional[int]
    cfo_hat_hz: Optional[float]
    pss_metric: Optional[float]

    # --- debug ---
    debug: Dict[str, Any]


# ======================================================================
# LTE System
# ======================================================================
class LTESystem:
    """
    Kompozitni LTE sistem (TX + Channel + RX).

    Klasa objedinjuje postojeće TX, Channel i RX module
    i omogućava jednostavno pokretanje kompletne
    end-to-end LTE simulacije jednim pozivom.

    Parameters
    ----------
    tx : LTETxChain
        LTE predajni lanac (OFDM modulacija, PSS, PBCH).

    ch : LTEChannel
        Kanal koji modelira frekvencijski ofset (CFO)
        i aditivni bijeli Gaussov šum (AWGN).

    rx : LTERxChain
        LTE prijemni lanac (PSS sinkronizacija,
        OFDM demodulacija, PBCH dekodiranje, CRC).
    """

    def __init__(self, tx, ch, rx) -> None:
        self.tx = tx
        self.ch = ch
        self.rx = rx

    # ------------------------------------------------------------------
    # Helper funkcije
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Resetuje interni state kanala (npr. brojač uzoraka za CFO).
        """
        if hasattr(self.ch, "reset"):
            self.ch.reset()

    @staticmethod
    def _to_bits_01(bits: Sequence[int]) -> np.ndarray:
        """
        Validira i konvertuje MIB bitove u niz 0/1.

        Parameters
        ----------
        bits : Sequence[int]
            Ulazni bitovi.

        Returns
        -------
        np.ndarray
            Niz dužine 24 sa vrijednostima 0 ili 1.
        """
        b = np.asarray(bits, dtype=np.uint8).flatten()
        if b.size != 24:
            raise ValueError(f"MIB mora imati 24 bita, dobio sam {b.size}.")
        return (b & 1).astype(np.uint8)

    @staticmethod
    def _compute_bit_errors(
        tx_bits: np.ndarray,
        rx_bits: Optional[np.ndarray]
    ) -> tuple[Optional[int], Optional[float]]:
        """
        Računa broj bitnih grešaka i BER.

        Returns
        -------
        bit_errors : int or None
        ber : float or None
        """
        if rx_bits is None:
            return None, None

        rx_bits = np.asarray(rx_bits, dtype=np.uint8).flatten()

        n = min(tx_bits.size, rx_bits.size)
        if n == 0:
            return None, None

        errors = int(np.sum(tx_bits[:n] != rx_bits[:n]))
        ber = errors / n
        return errors, ber

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def run(
        self,
        mib_bits: Sequence[int],
        reserved_re_mask: Optional[np.ndarray] = None,
        reset_channel: bool = True,
        keep_debug: bool = True,
    ) -> Dict[str, Any]:
        """
        Pokreće kompletnu end-to-end LTE simulaciju.

        Parameters
        ----------
        mib_bits : Sequence[int]
            MIB informacijski bitovi (24 bita).

        reserved_re_mask : np.ndarray, optional
            Bool maska rezervisanih resource elemenata
            za TX mapiranje.

        reset_channel : bool, optional
            Ako je True, resetuje kanal prije simulacije.

        keep_debug : bool, optional
            Ako je True, vraća detaljne RX debug informacije.

        Returns
        -------
        results : dict
            Rječnik sa signalima i metrikama sistema:
            - tx_waveform
            - rx_waveform
            - mib_bits_tx, mib_bits_rx
            - crc_ok
            - bit_errors, ber
            - detected_nid, tau_hat, cfo_hat_hz
            - pss_metric
            - debug
        """
        # --------------------------------------------------
        # Validacija MIB-a
        # --------------------------------------------------
        mib_tx = self._to_bits_01(mib_bits)

        if reset_channel:
            self.reset()

        # --------------------------------------------------
        # 1) TX
        # --------------------------------------------------
        tx_waveform, fs_hz = self.tx.generate_waveform(
            mib_bits=mib_tx,
            reserved_re_mask=reserved_re_mask
        )

        # --------------------------------------------------
        # 2) Channel
        # --------------------------------------------------
        rx_waveform = self.ch.apply(tx_waveform)

        # --------------------------------------------------
        # 3) RX
        # --------------------------------------------------
        rx_out = self.rx.process(rx_waveform)

        mib_rx = rx_out.get("mib_bits", None)
        crc_ok = bool(rx_out.get("crc_ok", False))
        debug_rx = rx_out.get("debug", {}) if keep_debug else {}

        # --------------------------------------------------
        # 4) Metrics
        # --------------------------------------------------
        bit_errors, ber = self._compute_bit_errors(mib_tx, mib_rx)

        detected_nid = debug_rx.get("detected_nid", None)
        tau_hat = debug_rx.get("tau_hat", None)
        cfo_hat = debug_rx.get("cfo_hat", None)

        pss_metric = None
        corr = debug_rx.get("pss_corr_metrics", None)
        if corr is not None:
            try:
                pss_metric = float(np.max(np.abs(np.asarray(corr))))
            except Exception:
                pass

        # --------------------------------------------------
        # Debug info
        # --------------------------------------------------
        system_debug = {
            "fs_hz": fs_hz,
            "tx_waveform_len": int(tx_waveform.size),
            "rx_waveform_len": int(rx_waveform.size),
        }
        if keep_debug:
            system_debug.update(debug_rx)

        # --------------------------------------------------
        # Results packing
        # --------------------------------------------------
        results = LTESystemResults(
            mib_bits_tx=mib_tx,
            mib_bits_rx=None if mib_rx is None else np.asarray(mib_rx, dtype=np.uint8),
            tx_waveform=np.asarray(tx_waveform),
            rx_waveform=np.asarray(rx_waveform),
            fs_hz=float(fs_hz),
            crc_ok=crc_ok,
            bit_errors=bit_errors,
            ber=ber,
            detected_nid=None if detected_nid is None else int(detected_nid),
            tau_hat=None if tau_hat is None else int(tau_hat),
            cfo_hat_hz=None if cfo_hat is None else float(cfo_hat),
            pss_metric=pss_metric,
            debug=system_debug,
        )

        return asdict(results)
