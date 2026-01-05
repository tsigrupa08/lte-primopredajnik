# receiver/pss_sync.py
from __future__ import annotations

import numpy as np

from transmitter.LTETxChain import LTETxChain
from transmitter.ofdm import OFDMModulator
from channel.frequency_offset import FrequencyOffset


class PSSSynchronizer:
    """
    PSSSynchronizer za LTE OFDM waveform (RX-only PSS sync + CFO).

    Ova klasa radi tri ključna koraka:
      1) PSS korelacija (3 kandidata) -> korelacijske metrike kroz vrijeme (tau)
      2) Timing + izbor N_ID_2        -> tau_hat, N_ID_2_hat
      3) CFO procjena + korekcija     -> cfo_hat, rx_corr

    Implementacija je usklađena s OFDM signalom koji generiše tvoj LTETxChain/OFDMModulator:
    - Korelacija se radi sa time-domain OFDM template-om PSS simbola (CP + N uzoraka),
      ne sa 62 ZC uzorka u vremenu.
    - CFO se procjenjuje CP-autokorelacijom:
        v = sum( tail * conj(cp) )
        phi = angle(v)
        cfo_hat = phi * Fs / (2*pi*N)
      gdje je "period" = N (FFT size), jer je CP kopija zadnjih cp_len uzoraka IFFT dijela.

    Parametri __init__:
    -------------------
    sample_rate_hz : float
        Sampling frekvencija RX signala (Hz). Mora se poklopiti s onom koju koristi TX (fs).
    n_id_2_candidates : tuple[int, ...]
        Kandidati za N_ID_2 (tipično (0,1,2)).
    ndlrb : int
        Broj DL resource block-ova (npr. 6 za LTE 1.4 MHz).
    normal_cp : bool
        True za normal CP, False za extended CP.

    Tipični tok korištenja:
    -----------------------
    >>> sync = PSSSynchronizer(sample_rate_hz=fs, ndlrb=6, normal_cp=True)
    >>> corr = sync.correlate(rx)                 # (3, corr_len) kompleksno
    >>> tau_hat, nid_hat = sync.estimate_timing(corr)
    >>> cfo_hat = sync.estimate_cfo(rx, tau_hat, nid_hat)
    >>> rx_corr = sync.apply_cfo_correction(rx, cfo_hat)

    Napomena (važna za demo):
    -------------------------
    CP CFO metoda ima praktični capture range ~ ±Fs/(2N).
    Za ndlrb=6: Fs=1.92 MHz, N=128 -> ±7.5 kHz.
    Za čistu vizualizaciju u C2 koristi CFO_true npr. 3000–7000 Hz.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        n_id_2_candidates: tuple[int, ...] = (0, 1, 2),
        ndlrb: int = 6,
        normal_cp: bool = True,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.n_id_2_candidates = tuple(n_id_2_candidates)
        self.ndlrb = int(ndlrb)
        self.normal_cp = bool(normal_cp)

        # Popunjava se pri gradnji template-a
        self._fft_N: int | None = None
        self._cp_len_pss: int | None = None

        # Template-i se grade jednom (CP+N PSS simbol u vremenu)
        self._templates = self._build_time_templates()

    @staticmethod
    def _symbol_start_indices(ofdm: OFDMModulator) -> list[int]:
        """
        Vraća start indekse OFDM simbola u vremenskom signalu (CP start).

        Returns
        -------
        starts : list[int]
            Lista start indeksa (u uzorcima) za svaki OFDM simbol u grid-u.
        """
        starts: list[int] = []
        idx = 0
        for sym_idx in range(ofdm.num_ofdm_symbols):
            starts.append(idx)
            cp_len = int(ofdm.cp_lengths[sym_idx % ofdm.n_symbols_per_slot])
            idx += ofdm.N + cp_len
        return starts

    def _build_time_templates(self) -> dict[int, np.ndarray]:
        """
        Generiše time-domain template PSS simbola (CP+N) za svaki N_ID_2 kandidat.

        Returns
        -------
        templates : dict[int, np.ndarray]
            Mapa N_ID_2 -> template (kompleksni niz dužine CP+N).

        Raises
        ------
        ValueError:
            Ako se sampling frekvencija template-a ne poklapa s sample_rate_hz.
        RuntimeError:
            Ako se ne uspije postaviti FFT N ili CP length (PSS).
        """
        templates: dict[int, np.ndarray] = {}

        for nid in self.n_id_2_candidates:
            tx = LTETxChain(
                n_id_2=int(nid),
                ndlrb=self.ndlrb,
                num_subframes=1,
                normal_cp=self.normal_cp,
            )
            tx_waveform, fs = tx.generate_waveform(mib_bits=None)

            fs = float(fs)
            if abs(fs - self.sample_rate_hz) > 1e-6:
                raise ValueError(f"Template fs={fs} Hz != RX fs={self.sample_rate_hz} Hz")

            ofdm = OFDMModulator(tx.grid)
            starts = self._symbol_start_indices(ofdm)

            # PSS simbol index: normal CP -> l=6, extended -> l=5 (u prvom slotu)
            pss_sym = 6 if self.normal_cp else 5
            pss_start = int(starts[pss_sym])

            cp_len = int(ofdm.cp_lengths[pss_sym % ofdm.n_symbols_per_slot])
            N = int(ofdm.N)
            L = int(N + cp_len)

            if self._fft_N is None:
                self._fft_N = N
            if self._cp_len_pss is None:
                self._cp_len_pss = cp_len

            templates[int(nid)] = np.asarray(tx_waveform[pss_start:pss_start + L], dtype=np.complex128)

        if self._fft_N is None or self._cp_len_pss is None:
            raise RuntimeError("Nije postavljen FFT N ili CP length (PSS).")

        return templates

    def correlate(self, rx_waveform: np.ndarray) -> np.ndarray:
        """
        Radi normalizovanu korelaciju RX signala sa PSS template-ima (CP+N).

        Parameters
        ----------
        rx_waveform : np.ndarray
            Kompleksni baznopojasni RX signal (1D).

        Returns
        -------
        corr_metrics : np.ndarray
            Kompleksne korelacijske metrike oblika (K, corr_len), gdje je:
              - K = broj kandidata (tipično 3),
              - corr_len = len(rx) - L + 1,
              - L = dužina template-a (CP+N).
            Obično se za odluku gleda |corr_metrics|.

        Raises
        ------
        ValueError:
            Ako je RX prekratak u odnosu na dužinu template-a.
        """
        rx = np.asarray(rx_waveform, dtype=np.complex128)

        nids = list(self._templates.keys())
        L = self._templates[nids[0]].size
        corr_len = rx.size - L + 1
        if corr_len <= 0:
            raise ValueError("RX je prekratak za PSS korelaciju (CP+N template).")

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

    def estimate_timing(self, corr_metrics: np.ndarray) -> tuple[int, int]:
        """
        Iz korelacijskih metrika bira maksimalni peak:
        - tau_hat: indeks uzorka gdje je peak (procjena tajminga)
        - N_ID_2_hat: kandidat koji je dao najveći peak

        Parameters
        ----------
        corr_metrics : np.ndarray
            Izlaz iz correlate(): oblik (K, corr_len), kompleksno.

        Returns
        -------
        tau_hat : int
            Procijenjeni timing offset (u uzorcima).
        N_ID_2_hat : int
            Procijenjeni N_ID_2 (0/1/2).
        """
        max_idx = np.unravel_index(np.abs(corr_metrics).argmax(), corr_metrics.shape)
        tau_hat = int(max_idx[1])
        detected_nid = int(self.n_id_2_candidates[int(max_idx[0])])
        return tau_hat, detected_nid

    def estimate_cfo(self, rx_waveform: np.ndarray, tau_hat: int, n_id_2: int) -> float:
        """
        CFO procjena iz cyclic prefix-a (CP) autokorelacijom.

        Parameters
        ----------
        rx_waveform : np.ndarray
            Kompleksni RX signal (1D).
        tau_hat : int
            Procijenjeni start PSS simbola (CP start) u RX signalu.
        n_id_2 : int
            Detektovani N_ID_2 (ovdje se ne koristi direktno u CP metodi,
            ali se čuva u potpisu radi konzistentnog API-ja).

        Returns
        -------
        cfo_hat : float
            Procijenjeni CFO u Hz.

        Raises
        ------
        ValueError:
            Ako RX nema dovoljno uzoraka za jedan OFDM simbol (CP+N) od tau_hat.
        """
        rx = np.asarray(rx_waveform, dtype=np.complex128)

        N = int(self._fft_N)
        cp = int(self._cp_len_pss)

        sym = rx[int(tau_hat): int(tau_hat) + (N + cp)]
        if sym.size < (N + cp):
            raise ValueError("RX segment za CFO (CP metoda) je prekratak.")

        cp_part = sym[0:cp]
        tail_part = sym[N:N + cp]

        v = np.sum(tail_part * np.conj(cp_part))
        phi = np.angle(v)
        cfo_hat = phi * self.sample_rate_hz / (2.0 * np.pi * N)
        return float(cfo_hat)

    def apply_cfo_correction(self, rx_waveform: np.ndarray, cfo_hat: float) -> np.ndarray:
        """
        Primjenjuje CFO korekciju na cijeli RX signal.

        Parameters
        ----------
        rx_waveform : np.ndarray
            Kompleksni RX signal (1D).
        cfo_hat : float
            Procijenjeni CFO u Hz. Primjenjuje se -cfo_hat.

        Returns
        -------
        rx_corr : np.ndarray
            CFO-korigovani RX signal.
        """
        rx = np.asarray(rx_waveform, dtype=np.complex128)
        fo = FrequencyOffset(freq_offset_hz=-float(cfo_hat), sample_rate_hz=self.sample_rate_hz)
        return fo.apply(rx)

