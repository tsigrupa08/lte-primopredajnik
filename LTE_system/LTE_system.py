from __future__ import annotations

import numpy as np
from typing import Sequence, Optional, Dict, Any

from transmitter.LTETxChain import LTETxChain
from channel.lte_channel import LTEChannel
from receiver.LTERxChain import LTERxChain


class LTESystem:
    """
    LTE end-to-end sistem (TX + kanal + RX).

    Ova klasa NE implementira DSP algoritme,
    već orkestrira postojeće module:

        TX → Channel → RX

    Namijenjena je za:
    - demo skripte
    - GUI
    - sistemske testove
    """

    def __init__(
        self,
        tx: LTETxChain,
        channel: LTEChannel,
        rx: LTERxChain,
    ) -> None:
        self.tx = tx
        self.channel = channel
        self.rx = rx

    def run(
        self,
        mib_bits: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Pokreće kompletan LTE sistem.

        Parameters
        ----------
        mib_bits : Sequence[int] or None
            24 MIB bita (0/1).
            Ako je None → TX šalje samo PSS.

        Returns
        -------
        dict
            {
                "tx": {
                    "waveform": np.ndarray,
                    "fs": float
                },
                "channel": {
                    "rx_waveform": np.ndarray
                },
                "rx": {
                    "mib_bits": np.ndarray,
                    "crc_ok": bool,
                    "debug": dict
                }
            }
        """

        # --------------------------------------------------
        # 1) TX
        # --------------------------------------------------
        tx_waveform, fs = self.tx.generate_waveform(mib_bits=mib_bits)

        # --------------------------------------------------
        # 2) Kanal
        # --------------------------------------------------
        self.channel.reset()
        rx_waveform = self.channel.apply(tx_waveform)

        # --------------------------------------------------
        # 3) RX
        # --------------------------------------------------
        rx_result = self.rx.process(rx_waveform)

        # --------------------------------------------------
        # 4) Spakovani rezultati
        # --------------------------------------------------
        return {
            "tx": {
                "waveform": tx_waveform,
                "fs": fs,
            },
            "channel": {
                "rx_waveform": rx_waveform,
            },
            "rx": rx_result,
        }
