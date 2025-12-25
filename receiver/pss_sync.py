import numpy as np
from transmitter.pss import PSSGenerator
from channel.frequency_offset import FrequencyOffset


class PSSSynchronizer:
    """
    PSSSynchronizer vrši obradu LTE Primary Synchronization Signal (PSS)
    u prijemniku.
    Modul obavlja:
    - korelaciju primljenog signala sa sve tri moguće PSS sekvence (N_ID_2 = 0, 1, 2)
    - detekciju ispravnog N_ID_2 indeksa
    - procjenu vremenskog pomaka (timing)
    - procjenu frekvencijskog ofseta (CFO)
    - korekciju CFO-a nad primljenim signalom

    Primjeri korištenja
    ------------------
    >>> import numpy as np
    >>> from rx.pss_sync import PSSSynchronizer
    >>>
    >>> # Simulacija primljenog kompleksnog signala
    >>> rx_waveform = np.random.randn(1024) + 1j * np.random.randn(1024)
    >>> # Kreiranje PSSSynchronizer objekta
    >>> pss_sync = PSSSynchronizer(sample_rate_hz=1e6)
    >>>
    >>> # Korelacija sa svim PSS kandidatima
    >>> corr_metrics = pss_sync.correlate(rx_waveform)
    >>> # Prikaz magnitude korelacije za sve 3 PSS sekvence
    >>> import matplotlib.pyplot as plt
    >>> for i, nid in enumerate(pss_sync.n_id_2_candidates):
    ...     plt.plot(np.abs(corr_metrics[i]), label=f"N_ID_2={nid}")
    >>> plt.xlabel("Uzorak")
    >>> plt.ylabel("Magnitude korelacije")
    >>> plt.legend()
    >>> plt.show()
    >>>
    >>> # Procjena vremenskog pomaka i detekcija N_ID_2
    >>> tau_hat, detected_nid = pss_sync.estimate_timing(corr_metrics)
    >>> print("Procijenjeni tau:", tau_hat)
    >>> print("Detektovani N_ID_2:", detected_nid)
    >>>
    >>> # Procjena frekvencijskog ofseta (CFO)
    >>> cfo_hat = pss_sync.estimate_cfo(rx_waveform, tau_hat, detected_nid)
    >>> print("Procijenjeni CFO (Hz):", cfo_hat)
    >>>
    >>> # Korekcija CFO-a na primljeni signal
    >>> rx_corrected = pss_sync.apply_cfo_correction(rx_waveform, cfo_hat)

    Napomene
    --------
    - Klasa je namijenjena za LTE prijemne lance i sinhronizaciju PSS signala.
    - Funkcije koriste kompleksne numpy nizove i NumPy operacije za vektorizaciju.
    - Primjeri korištenja pokazuju puni lanac: korelacija, detekcija N_ID_2, procjena tau i CFO, korekcija signala.
    - Prikaz magnitude korelacije pomaže vizualno provjeriti koji N_ID_2 signal najbolje odgovara primljenom signalu.
    """

    def __init__(self, sample_rate_hz, n_id_2_candidates=(0, 1, 2)):
        """
        Parametri
        ----------
        sample_rate_hz : float
            Sample rate primljenog signala u Hz.
        n_id_2_candidates : tuple of int
            Kandidati za N_ID_2 (podrazumijevano: (0, 1, 2)).
        """
        self.sample_rate_hz = float(sample_rate_hz)
        self.n_id_2_candidates = n_id_2_candidates

    def correlate(self, rx_waveform):
        """
        Vrši korelaciju primljenog signala sa svim PSS kandidatima.

        Parametri
        ----------
        rx_waveform : np.ndarray
            1D kompleksni baznopojasni primljeni signal.

        Povratna vrijednost
        -------------------
        corr_metrics : np.ndarray
            Kompleksne korelacione metrike oblika
            (broj_kandidata, vrijeme).

        Komentari
        ---------
        - Korelacija u kompleksnom domenu zahtijeva konjugovanu referencu.
        - Vektorizirano za sve tri PSS sekvence.
        """
        rx_waveform = np.asarray(rx_waveform)
        pss_len = len(PSSGenerator.generate(self.n_id_2_candidates[0]))
        corr_len = len(rx_waveform) - pss_len + 1

        corr_metrics = np.zeros(
            (len(self.n_id_2_candidates), corr_len),
            dtype=np.complex64
        )

        for i, nid in enumerate(self.n_id_2_candidates):
            pss = PSSGenerator.generate(nid)
            corr = np.correlate(rx_waveform, np.conj(pss), mode="valid")
            corr_metrics[i, :] = corr

        return corr_metrics

    def estimate_timing(self, corr_metrics):
        """
        Procjenjuje vremenski pomak (tau_hat) i detektuje N_ID_2
        na osnovu maksimuma magnitude korelacije.

        Parametri
        ----------
        corr_metrics : np.ndarray
            Korelacione metrike dobijene iz metode `correlate()`.

        Povratne vrijednosti
        --------------------
        tau_hat : int
            Procijenjeni vremenski pomak (indeks uzorka).
        detected_nid : int
            Detektovani N_ID_2 indeks.

        Komentari
        ---------
        - Maksimalna magnitude korelacije pokazuje najvjerovatniji PSS signal.
        """
        max_idx = np.unravel_index(
            np.abs(corr_metrics).argmax(),
            corr_metrics.shape
        )

        tau_hat = int(max_idx[1])
        detected_nid = self.n_id_2_candidates[max_idx[0]]

        return tau_hat, detected_nid

    def estimate_cfo(self, rx_waveform, tau_hat, n_id_2):
        """
        Procjenjuje frekvencijski ofset (CFO) iz detektovanog PSS segmenta.

        Parametri
        ----------
        rx_waveform : np.ndarray
            Primljeni kompleksni signal.
        tau_hat : int
            Procijenjeni vremenski pomak.
        n_id_2 : int
            Detektovani PSS indeks.

        Povratna vrijednost
        -------------------
        cfo_hat : float
            Procijenjeni CFO u Hz.

        Komentari
        ---------
        - Procjena se bazira na prosječnoj faznoj rotaciji između uzastopnih uzoraka PSS-a.
        """
        pss = PSSGenerator.generate(n_id_2)
        pss_len = len(pss)
        rx_pss = rx_waveform[tau_hat:tau_hat + pss_len]

        phase_diff = np.angle(rx_pss[1:] * np.conj(rx_pss[:-1]))
        cfo_hat = np.mean(phase_diff) * self.sample_rate_hz / (2.0 * np.pi)

        return float(cfo_hat)

    def apply_cfo_correction(self, rx_waveform, cfo_hat):
        """
        Primjenjuje korekciju frekvencijskog ofseta (CFO) na primljeni signal.

        Parametri
        ----------
        rx_waveform : np.ndarray
            Primljeni kompleksni signal.
        cfo_hat : float
            Procijenjeni CFO u Hz.

        Povratna vrijednost
        -------------------
        rx_corrected : np.ndarray
            Signal nakon korekcije CFO-a.

        Komentari
        ---------
        - Negativni ofset se primjenjuje da bi se poništio procijenjeni CFO.
        """
        fo = FrequencyOffset(
            freq_offset_hz=-cfo_hat,
            sample_rate_hz=self.sample_rate_hz
        )
        rx_corrected = fo.apply(rx_waveform)
        return rx_corrected
