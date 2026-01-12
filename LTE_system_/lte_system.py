"""
lte_system.py
=============

End-to-end LTE sistem koji povezuje kompletan predajni i prijemni lanac:

    TX (LTETxChain) → Channel (LTEChannel) → RX (LTERxChain)

Modul služi kao “glue layer” između TX, Channel i RX blokova i
namijenjen je demonstraciji rada LTE sistema u cjelini
(kroz demo skripte i GUI).

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
from typing import Any, Dict, Optional, Sequence, Tuple

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
    detected_nid: Optional[int]      # ovdje je to N_ID_2_hat (u tvojoj implementaciji)
    tau_hat: Optional[int]
    cfo_hat_hz: Optional[float]
    pss_metric: Optional[float]      # npr. corr_peak

    # --- debug ---
    debug: Dict[str, Any]


# ======================================================================
# LTE System
# ======================================================================
class LTESystem:
    """
    Kompozitni LTE sistem (TX + Channel + RX).
    """

    def __init__(self, tx, ch, rx) -> None:
        self.tx = tx
        self.ch = ch
        self.rx = rx

    # ------------------------------------------------------------------
    # Helper funkcije
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Resetuje interni state kanala (npr. brojač uzoraka za CFO)."""
        if hasattr(self.ch, "reset"):
            self.ch.reset()

    @staticmethod
    def _to_bits_01(bits: Sequence[int]) -> np.ndarray:
        """Validira i konvertuje MIB bitove u niz 0/1 dužine 24."""
        b = np.asarray(bits, dtype=np.uint8).flatten()
        if b.size != 24:
            raise ValueError(f"MIB mora imati 24 bita, dobio sam {b.size}.")
        return (b & 1).astype(np.uint8)

    @staticmethod
    def _compute_bit_errors(
        tx_bits: np.ndarray,
        rx_bits: Optional[np.ndarray],
        count_all_if_missing: bool = True,
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Računa broj bitnih grešaka i BER.

        Ako rx_bits ne postoji (npr. CRC fail), možeš:
          - count_all_if_missing=True  -> tretiraj kao 24/24 grešaka (BER=1)
          - count_all_if_missing=False -> vrati None, None
        """
        if rx_bits is None:
            if count_all_if_missing:
                n = int(tx_bits.size)
                return n, 1.0
            return None, None

        rx_bits = np.asarray(rx_bits, dtype=np.uint8).flatten()
        n = min(int(tx_bits.size), int(rx_bits.size))
        if n <= 0:
            return None, None

        errors = int(np.sum(tx_bits[:n] != rx_bits[:n]))
        ber = float(errors) / float(n)
        return errors, ber

    @staticmethod
    def _rx_unpack(rx_out: Any) -> Tuple[Optional[np.ndarray], bool, Optional[int], Optional[int], Optional[float], Dict[str, Any]]:
        """
        Podržava:
          - RxResult dataclass (rx_out.mib_bits, rx_out.crc_ok, rx_out.n_id_2_hat, rx_out.tau_hat, rx_out.cfo_hat, rx_out.debug)
          - dict (ključevi 'mib_bits', 'crc_ok', ...)
        """
        # RxResult dataclass-like
        if hasattr(rx_out, "crc_ok") and hasattr(rx_out, "debug"):
            mib = getattr(rx_out, "mib_bits", None)
            crc_ok = bool(getattr(rx_out, "crc_ok", False))
            n_id_2_hat = getattr(rx_out, "n_id_2_hat", None)
            tau_hat = getattr(rx_out, "tau_hat", None)
            cfo_hat = getattr(rx_out, "cfo_hat", None)
            dbg = getattr(rx_out, "debug", {}) or {}
            return mib, crc_ok, n_id_2_hat, tau_hat, cfo_hat, dbg

        # dict-like
        if isinstance(rx_out, dict):
            mib = rx_out.get("mib_bits", None)
            crc_ok = bool(rx_out.get("crc_ok", False))
            dbg = rx_out.get("debug", {}) or {}
            n_id_2_hat = rx_out.get("n_id_2_hat", dbg.get("n_id_2_hat", None))
            tau_hat = rx_out.get("tau_hat", dbg.get("tau_hat", None))
            cfo_hat = rx_out.get("cfo_hat", dbg.get("cfo_hat_hz", None))
            return mib, crc_ok, n_id_2_hat, tau_hat, cfo_hat, dbg

        return None, False, None, None, None, {}

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def run(
        self,
        mib_bits: Sequence[int],
        reserved_re_mask: Optional[np.ndarray] = None,
        reset_channel: bool = True,
        keep_debug: bool = True,
        count_all_if_rx_missing: bool = True,
    ) -> Dict[str, Any]:
        """
        Pokreće kompletnu end-to-end LTE simulaciju.
        """
        # 0) Validacija MIB-a
        mib_tx = self._to_bits_01(mib_bits)

        if reset_channel:
            self.reset()

        # 1) TX
        tx_waveform, fs_hz = self.tx.generate_waveform(
            mib_bits=mib_tx,
            reserved_re_mask=reserved_re_mask,
        )

        # 2) Channel
        rx_waveform = self.ch.apply(tx_waveform)

        # 3) RX (decode je primarni API u tvom LTERxChain)
        if hasattr(self.rx, "decode"):
            rx_raw = self.rx.decode(rx_waveform)
        elif hasattr(self.rx, "process"):
            rx_raw = self.rx.process(rx_waveform)
        else:
            raise AttributeError("RX objekt nema ni decode() ni process().")

        mib_rx, crc_ok, n_id_2_hat, tau_hat, cfo_hat, debug_rx = self._rx_unpack(rx_raw)
        debug_rx = debug_rx if keep_debug else {}

        # 4) Metrics
        bit_errors, ber = self._compute_bit_errors(
            mib_tx,
            None if (mib_rx is None) else np.asarray(mib_rx, dtype=np.uint8),
            count_all_if_missing=count_all_if_rx_missing,
        )

        # PSS “metric”: kod tebe je najkorisnije corr_peak (iz debug-a)
        pss_metric = None
        if keep_debug:
            if "corr_peak" in debug_rx:
                try:
                    pss_metric = float(debug_rx["corr_peak"])
                except Exception:
                    pss_metric = None

        # 5) System debug
        system_debug: Dict[str, Any] = {
            "fs_hz": float(fs_hz),
            "tx_waveform_len": int(np.asarray(tx_waveform).size),
            "rx_waveform_len": int(np.asarray(rx_waveform).size),
        }
        if keep_debug:
            system_debug.update(debug_rx)

        # 6) Results packing
        results = LTESystemResults(
            mib_bits_tx=np.asarray(mib_tx, dtype=np.uint8),
            mib_bits_rx=None if mib_rx is None else np.asarray(mib_rx, dtype=np.uint8),
            tx_waveform=np.asarray(tx_waveform),
            rx_waveform=np.asarray(rx_waveform),
            fs_hz=float(fs_hz),
            crc_ok=bool(crc_ok),
            bit_errors=bit_errors,
            ber=ber,
            detected_nid=None if n_id_2_hat is None else int(n_id_2_hat),
            tau_hat=None if tau_hat is None else int(tau_hat),
            cfo_hat_hz=None if cfo_hat is None else float(cfo_hat),
            pss_metric=pss_metric,
            debug=system_debug,
        )

        return asdict(results)
