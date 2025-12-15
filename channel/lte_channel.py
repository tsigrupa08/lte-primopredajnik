import numpy as np
from .frequency_offset import FrequencyOffset
from .awgn_channel import AWGNChannel


class LTEChannel:
    """
    Kompozitni LTE kanal koji spaja dva osnovna impairments-a:
    frekvencijski ofset i AWGN šum.

    Kanal radi u slijedu:

        x → FrequencyOffset → AWGN → y

    Parameters
    ----------
    freq_offset_hz : float
        Frekvencijski ofset (Δf) u Hz.
    sample_rate_hz : float
        Sample rate signala u Hz.
    snr_db : float
        Ciljani odnos signal-šum (SNR) u decibelima.
    seed : int or None, optional
        Sjeme za generator slučajnih brojeva (AWGN). Default je ``None``.
    initial_phase_rad : float, optional
        Početna faza kompleksnog eksponencijala. Default je ``0.0``.
    """

    def __init__(
        self,
        freq_offset_hz: float,
        sample_rate_hz: float,
        snr_db: float,
        seed: int | None = None,
        initial_phase_rad: float = 0.0,
    ) -> None:

        self.fo = FrequencyOffset(
            freq_offset_hz=freq_offset_hz,
            sample_rate_hz=sample_rate_hz,
            initial_phase_rad=initial_phase_rad,
        )

        self.awgn = AWGNChannel(snr_db=snr_db, seed=seed)

    def reset(self) -> None:
        """
        Resetuje interni state FrequencyOffset modula (brojač uzoraka).

        Returns
        -------
        None
        """
        self.fo.reset()

    def apply(self, tx_samples: np.ndarray) -> np.ndarray:
        """
        Primjenjuje kompozitni LTE kanal nad ulaznim kompleksnim signalom.

        Redoslijed obrade:
        1) Frekvencijski ofset (rotacija kompleksnog signala)
        2) AWGN šum

        Parameters
        ----------
        tx_samples : np.ndarray
            Kompleksni NumPy niz sa uzorcima signala. Zadnja os je vremenski
            indeks. Može biti 1D, 2D ili višedimenzionalan.

        Returns
        -------
        np.ndarray
            Niz iste dimenzije kao `tx_samples`, nakon prolaska kroz kanal.

        Raises
        ------
        ValueError
            Ako `tx_samples` nije kompleksan NumPy niz.
        """
        # 1. Frekvencijski ofset
        y = self.fo.apply(tx_samples)

        # 2. AWGN šum
        y = self.awgn.apply(y)

        return y
