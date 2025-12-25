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

    Korištenje:
        - Instancira se sa sample_rate_hz i opcionalnim n_id_2_candidates
        - Nad instancom se pozivaju metode: correlate, estimate_timing,
          estimate_cfo i apply_cfo_correction
    """

    def __init__(self, sample_rate_hz, n_id_2_candidates=(0, 1, 2)):
        """
        Inicijalizuje PSSSynchronizer.

        Parametri:
            sample_rate_hz (float): Sample rate primljenog signala u Hz.
            n_id_2_candidates (tuple of int): Kandidati za N_ID_2 (podrazumijevano: (0,1,2)).

        Korištenje:
            - Kreira instancu klase koja može obrađivati primljeni PSS signal
        """
        self.sample_rate_hz = float(sample_rate_hz)
        self.n_id_2_candidates = n_id_2_candidates

    def correlate(self, rx_waveform):
        """
        Korelacija primljenog signala sa svim PSS kandidatima.

        Parametri:
            rx_waveform (np.ndarray): 1D kompleksni baznopojasni primljeni signal.

        Povratna vrijednost:
            np.ndarray: Kompleksne korelacione metrike oblika
                        (broj_kandidata, vrijeme).

        Korištenje:
            - Metoda se poziva nad instancom PSSSynchronizer
            - Prima primljeni signal i vraća korelacione metrike
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
        Procjena vremenskog pomaka i detekcija N_ID_2.

        Parametri:
            corr_metrics (np.ndarray): Korelacione metrike iz metode correlate().

        Povratne vrijednosti:
            tau_hat (int): Procijenjeni vremenski pomak (indeks uzorka).
            detected_nid (int): Detektovani N_ID_2 indeks.

        Korištenje:
            - Poziva se nad instancom PSSSynchronizer
            - Prima korelacione metrike i vraća procijenjeni pomak i N_ID_2
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
        Procjena frekvencijskog ofseta (CFO).

        Parametri:
            rx_waveform (np.ndarray): Primljeni kompleksni signal.
            tau_hat (int): Procijenjeni vremenski pomak.
            n_id_2 (int): Detektovani PSS indeks.

        Povratna vrijednost:
            cfo_hat (float): Procijenjeni CFO u Hz.

        Korištenje:
            - Poziva se nad instancom PSSSynchronizer
            - Prima signal, procijenjeni pomak i N_ID_2, vraća CFO
        """
        pss = PSSGenerator.generate(n_id_2)
        pss_len = len(pss)
        rx_pss = rx_waveform[tau_hat:tau_hat + pss_len]
        phase_diff = np.angle(rx_pss[1:] * np.conj(rx_pss[:-1]))
        cfo_hat = np.mean(phase_diff) * self.sample_rate_hz / (2.0 * np.pi)
        return float(cfo_hat)

    def apply_cfo_correction(self, rx_waveform, cfo_hat):
        """
        Primjena korekcije frekvencijskog ofseta (CFO) na primljeni signal.

        Parametri:
            rx_waveform (np.ndarray): Primljeni kompleksni signal.
            cfo_hat (float): Procijenjeni CFO u Hz.

        Povratna vrijednost:
            rx_corrected (np.ndarray): Signal nakon korekcije CFO-a.

        Korištenje:
            - Poziva se nad instancom PSSSynchronizer
            - Prima signal i procijenjeni CFO, vraća korektovan signal
        """
        fo = FrequencyOffset(
            freq_offset_hz=-cfo_hat,
            sample_rate_hz=self.sample_rate_hz
        )
        rx_corrected = fo.apply(rx_waveform)
        return rx_corrected
