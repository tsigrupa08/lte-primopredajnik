import numpy as np

from transmitter.LTETxChain import LTETxChain
from transmitter.ofdm import OFDMModulator
from channel.frequency_offset import FrequencyOffset


class PSSSynchronizer:
    """
    PSSSynchronizer (fix za OFDM waveform):
    - Korelacija radi sa time-domain PSS TEMPLATE-om (CP+N uzoraka),
      a ne sa 62 ZC uzorka u vremenu.
    - CFO procjena radi iz z[n] = rx_seg[n] * conj(template[n]),
      pa fazni nagib daje CFO.
    """

    def __init__(self, sample_rate_hz, n_id_2_candidates=(0, 1, 2), ndlrb=6, normal_cp=True):
        self.sample_rate_hz = float(sample_rate_hz)
        self.n_id_2_candidates = tuple(n_id_2_candidates)
        self.ndlrb = int(ndlrb)
        self.normal_cp = bool(normal_cp)

        # Template-i se grade jednom
        self._templates = self._build_time_templates()

    @staticmethod
    def _symbol_start_indices(ofdm: OFDMModulator) -> list[int]:
        starts = []
        idx = 0
        for sym_idx in range(ofdm.num_ofdm_symbols):
            starts.append(idx)
            cp_len = int(ofdm.cp_lengths[sym_idx % ofdm.n_symbols_per_slot])
            idx += ofdm.N + cp_len
        return starts

    def _build_time_templates(self) -> dict[int, np.ndarray]:
        """
        Za svaki N_ID_2 generiše TX waveform (1 subframe) i izvadi PSS OFDM simbol (CP+N).
        """
        templates: dict[int, np.ndarray] = {}

        for nid in self.n_id_2_candidates:
            tx = LTETxChain(n_id_2=nid, ndlrb=self.ndlrb, num_subframes=1, normal_cp=self.normal_cp)
            tx_waveform, fs = tx.generate_waveform(mib_bits=None)

            # Sample-rate check (mora se poklapati s RX fs)
            fs = float(fs)
            if abs(fs - self.sample_rate_hz) > 1e-6:
                raise ValueError(f"Template fs={fs} Hz != RX fs={self.sample_rate_hz} Hz")

            ofdm = OFDMModulator(tx.grid)
            starts = self._symbol_start_indices(ofdm)

            pss_sym = 6 if self.normal_cp else 5
            pss_start = int(starts[pss_sym])
            cp_len = int(ofdm.cp_lengths[pss_sym % ofdm.n_symbols_per_slot])
            L = int(ofdm.N + cp_len)

            templates[nid] = np.asarray(tx_waveform[pss_start:pss_start + L], dtype=np.complex128)

        return templates

    def correlate(self, rx_waveform):
        """
        Normalizovana korelacija rx sa OFDM time-domain template-ima (CP+N):
            c[tau] = <rx_win, template> / (||rx_win|| * ||template||)
        Vraća kompleksne metrike (kandidati x tau).
        """
        rx = np.asarray(rx_waveform, dtype=np.complex128)

        nids = list(self._templates.keys())
        L = self._templates[nids[0]].size
        corr_len = rx.size - L + 1
        if corr_len <= 0:
            raise ValueError("RX je prekratak za PSS korelaciju (CP+N template).")

        # sliding energija rx prozora
        rx_energy = np.convolve(np.abs(rx) ** 2, np.ones(L, dtype=np.float64), mode="valid")
        rx_energy = np.maximum(rx_energy, 1e-12)

        corr_metrics = np.zeros((len(nids), corr_len), dtype=np.complex128)

        for i, nid in enumerate(nids):
            t = self._templates[nid]
            t_energy = float(np.sum(np.abs(t) ** 2))
            t_energy = max(t_energy, 1e-12)

            # np.correlate za kompleksne već radi conj(t) interno
            c = np.correlate(rx, t, mode="valid")
            corr_metrics[i, :] = c / np.sqrt(rx_energy * t_energy)

        return corr_metrics

    def estimate_timing(self, corr_metrics):
        """
        tau_hat + detected_nid iz maksimuma |corr|.
        """
        max_idx = np.unravel_index(np.abs(corr_metrics).argmax(), corr_metrics.shape)
        tau_hat = int(max_idx[1])
        detected_nid = self.n_id_2_candidates[max_idx[0]]
        return tau_hat, detected_nid

    def estimate_cfo(self, rx_waveform, tau_hat, n_id_2):
        """
        CFO iz segmenta oko PSS:
        - Uzmi rx_seg (CP+N) i template (CP+N)
        - z[n] = rx_seg[n] * conj(template[n])
        - CFO ~ mean(angle(z[n] * conj(z[n-1]))) * fs/(2pi)
        """
        rx = np.asarray(rx_waveform, dtype=np.complex128)
        t = self._templates[int(n_id_2)]
        L = t.size

        rx_seg = rx[int(tau_hat): int(tau_hat) + L]
        if rx_seg.size < L:
            raise ValueError("RX segment za CFO je prekratak.")

        z = rx_seg * np.conj(t)
        v = np.sum(z[1:] * np.conj(z[:-1]))       # kompleksni “average phasor”
        phi = np.angle(v)
        cfo_hat = phi * self.sample_rate_hz / (2.0 * np.pi)
        return float(cfo_hat)

    def apply_cfo_correction(self, rx_waveform, cfo_hat):
        """
        Primijeni -cfo_hat da poništi CFO.
        """
        rx = np.asarray(rx_waveform, dtype=np.complex128)
        fo = FrequencyOffset(freq_offset_hz=-cfo_hat, sample_rate_hz=self.sample_rate_hz)
        return fo.apply(rx)
