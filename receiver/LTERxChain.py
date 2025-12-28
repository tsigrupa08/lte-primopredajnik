import numpy as np

from receiver.utils import RxUtils
from receiver.pss_sync import PSSSynchronizer
from receiver.OFDM_demodulator import OFDMDemodulator
from receiver.resource_grid_extractor import PBCHExtractor, PBCHConfig
from receiver.QPSK_demapiranje import QPSKDemapper
from receiver.de_rate_matching import DeRateMatcher
from receiver.viterbi_decoder import ViterbiDecoder
from receiver.crc_checker import CRCChecker


class LTERxChain:
    """
    LTE Receiver Chain – PBCH prijemnik (RX).

    Ova klasa implementira kompletan LTE prijemni lanac za:
    - PSS sinkronizaciju (detekcija N_ID_2 i početka okvira)
    - OFDM demodulaciju (CP removal + FFT)
    - PBCH ekstrakciju i dekodiranje
    - CRC provjeru ispravnosti MIB poruke

    Klasa je dizajnirana za end-to-end demonstraciju:
    TX → Channel → RX

    Parameters
    ----------
    sample_rate_hz : float, optional
        Frekvencija uzorkovanja ulaznog signala (Hz).
        Tipično 1.92e6 za LTE 1.4 MHz.
    ndlrb : int, optional
        Broj downlink resource blokova (NDLRB).
        Za LTE 1.4 MHz vrijedi ndlrb = 6.
    normal_cp : bool, optional
        Ako je True, koristi se normalni cyclic prefix (14 OFDM simbola).
    fft_size : int or None, optional
        Ručno zadavanje FFT veličine (ako None, koristi se LTE standard).

    Notes
    -----
    Ova implementacija je edukativna i pojednostavljena,
    ali zadržava strukturu realnog LTE prijemnika.
    """

    def __init__(
        self,
        sample_rate_hz=1.92e6,
        ndlrb=6,
        normal_cp=True,
        fft_size=None
    ):
        # --------------------------------------------------
        # Pomoćne RX funkcije (normalizacija, validacija)
        # --------------------------------------------------
        self.utils = RxUtils()

        self.sample_rate_hz = float(sample_rate_hz)
        self.ndlrb = int(ndlrb)
        self.normal_cp = bool(normal_cp)

        # --------------------------------------------------
        # PSS sinkronizacija
        # --------------------------------------------------
        self.pss_sync = PSSSynchronizer(
            sample_rate_hz=self.sample_rate_hz
        )

        # --------------------------------------------------
        # OFDM demodulator (CP removal + FFT)
        # --------------------------------------------------
        self.ofdm_demod = OFDMDemodulator(
            ndlrb=self.ndlrb,
            normal_cp=self.normal_cp,
            new_fft_size=fft_size
        )

        # --------------------------------------------------
        # PBCH konfiguracija i ekstraktor
        # --------------------------------------------------
        pbch_cfg = PBCHConfig(
            ndlrb=self.ndlrb,
            normal_cp=self.normal_cp
        )
        self.pbch_ext = PBCHExtractor(pbch_cfg)

        # --------------------------------------------------
        # QPSK demapper (hard decision)
        # --------------------------------------------------
        self.demapper = QPSKDemapper(mode="hard")

        # --------------------------------------------------
        # De-rate matching (PBCH)
        # --------------------------------------------------
        self.deratematcher = DeRateMatcher(
            E_rx=1920,
            N_coded=120
        )

        # --------------------------------------------------
        # Viterbi dekoder (konvolucijski kod)
        # --------------------------------------------------
        self.viterbi = ViterbiDecoder(
            constraint_len=7,
            generators=[0o133, 0o171, 0o164],
            rate=1/3
        )

        # --------------------------------------------------
        # CRC provjera (MIB)
        # --------------------------------------------------
        self.crc = CRCChecker(
            poly=0x1021,
            init=0xFFFF
        )

    # ======================================================
    # RX PROCESS
    # ======================================================
    def process(self, rx_waveform: np.ndarray) -> dict:
        """
        Pokreće kompletan LTE RX lanac nad ulaznim signalom.

        Parameters
        ----------
        rx_waveform : np.ndarray (complex)
            Kompleksni vremenski signal na ulazu prijemnika
            (nakon kanala: AWGN, CFO, itd.).

        Returns
        -------
        result : dict
            Rječnik sa sljedećim ključevima:

            - 'mib_bits' : np.ndarray
                Dekodirani MIB bitovi (payload bez CRC).
            - 'crc_ok' : bool
                True ako je CRC ispravan, False inače.
            - 'debug' : dict
                Debug informacije korisne za analizu i vizualizaciju:
                * pss_corr_metrics
                * tau_hat
                * detected_nid
                * cfo_hat
                * grid_active
                * pbch_symbols_rx

        Notes
        -----
        Ova funkcija se tipično poziva jednom po RX signalu:

        >>> rx = LTERxChain(sample_rate_hz=fs)
        >>> out = rx.process(rx_waveform)
        >>> print(out["crc_ok"])
        """

        debug = {}

        # --------------------------------------------------
        # 1) Validacija i normalizacija signala
        # --------------------------------------------------
        rx_waveform = self.utils.validate_rx_samples(
            rx_waveform,
            min_num_samples=128
        )

        rx_waveform, scale = self.utils.normalize_rms(
            rx_waveform,
            target_rms=1.0
        )
        debug["rms_scale"] = scale

        # --------------------------------------------------
        # 2) PSS sinkronizacija
        #    - korelacija
        #    - detekcija timinga (tau_hat)
        #    - detekcija N_ID_2
        #    - procjena CFO
        # --------------------------------------------------
        corr = self.pss_sync.correlate(rx_waveform)
        tau_hat, detected_nid = self.pss_sync.estimate_timing(corr)
        cfo_hat = self.pss_sync.estimate_cfo(
            rx_waveform,
            tau_hat,
            detected_nid
        )

        debug["pss_corr_metrics"] = corr
        debug["tau_hat"] = tau_hat
        debug["detected_nid"] = detected_nid
        debug["cfo_hat"] = cfo_hat

        # --------------------------------------------------
        # 3) CFO korekcija
        # --------------------------------------------------
        rx_corr = self.pss_sync.apply_cfo_correction(
            rx_waveform,
            cfo_hat
        )

        # --------------------------------------------------
        # 4) OFDM demodulacija (CP removal + FFT)
        # --------------------------------------------------
        grid_fft = self.ofdm_demod.demodulate(rx_corr)
        # grid_fft shape: (N_OFDM_sym, FFT_size)

        # --------------------------------------------------
        # 5) Izdvajanje aktivnih subcarrier-a (72)
        # --------------------------------------------------
        active = self.ofdm_demod.extract_active_subcarriers(grid_fft)
        # active shape: (N_OFDM_sym, 72)

        # --------------------------------------------------
        # 6) Transpozicija za PBCH ekstraktor
        # --------------------------------------------------
        grid_pbch = active.T
        # grid_pbch shape: (72, N_OFDM_sym)

        debug["grid_active"] = grid_pbch

        # --------------------------------------------------
        # 7) PBCH ekstrakcija
        # --------------------------------------------------
        pbch_symbols_rx = self.pbch_ext.extract(grid_pbch)
        debug["pbch_symbols_rx"] = pbch_symbols_rx

        # --------------------------------------------------
        # 8) QPSK demapiranje
        # --------------------------------------------------
        demapped_bits = self.demapper.demap(pbch_symbols_rx)

        # --------------------------------------------------
        # 9) De-rate matching
        # --------------------------------------------------
        deratematched_bits = self.deratematcher.accumulate(
            demapped_bits,
            soft=False
        )

        # --------------------------------------------------
        # 10) Viterbi dekodiranje
        # --------------------------------------------------
        decoded_bits = self.viterbi.decode(deratematched_bits)

        # --------------------------------------------------
        # 11) CRC provjera (MIB)
        # --------------------------------------------------
        payload_bits, crc_ok = self.crc.check(decoded_bits)

        return {
            "mib_bits": payload_bits,
            "crc_ok": crc_ok,
            "debug": debug
        }
