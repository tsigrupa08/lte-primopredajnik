import numpy as np

# Import svih modula 
from receiver.utils import RxUtils
from receiver.pss_sync import PSSSynchronizer
from receiver.OFDM_demodulator import OFDMDemodulator
from receiver.resource_grid_extractor import PBCHExtractor
from receiver.QPSK_demapiranje import QPSKDemapper
from receiver.de_rate_matching import DeRateMatcher
from receiver.viterbi_decoder import ViterbiDecoder
from receiver.crc_checker import CRCChecker


class LTERxChain:
    """
    LTE Receiver Chain (PBCH prijemnik).
    Spaja sve korake prijema u tačnom redoslijedu:
    1. Validacija i normalizacija signala
    2. PSS korelacija → detekcija N_ID_2, timing, CFO
    3. CFO korekcija
    4. OFDM demodulacija (CP remove + FFT) + selekcija aktivnih podnosača
    5. Ekstrakcija PBCH simbola iz ACTIVE grida
    6. QPSK demapiranje u bitove
    7. De-rate-matching (akumulacija)
    8. Viterbi dekodiranje
    9. CRC provjera → vraća payload (MIB) i CRC flag

    Rezultat je dict sa svim međukoracima (debug) i finalnim MIB bitovima.
    """

    def __init__(self,
                 sample_rate_hz=1.92e6,  # 128 * 15 kHz for NDLRB=6
                 ndlrb=6,
                 normal_cp=True,
                 fft_size=None):
        # Utils za validaciju i RMS normalizaciju
        self.utils = RxUtils()

        # Konfiguracija
        self.sample_rate_hz = sample_rate_hz
        self.ndlrb = int(ndlrb)
        self.normal_cp = bool(normal_cp)

        # PSS sync
        self.pss_sync = PSSSynchronizer(sample_rate_hz=sample_rate_hz)

        # OFDM demodulator
        self.ofdm_demod = OFDMDemodulator(ndlrb=self.ndlrb,
                                          normal_cp=self.normal_cp,
                                          new_fft_size=fft_size)

        # PBCH extractor (inicijalizovan sa istom numerologijom)
        self.pbch_ext = PBCHExtractor(ndlrb=self.ndlrb, normal_cp=self.normal_cp)

        # QPSK demapper
        self.demapper = QPSKDemapper(mode="hard")

        # De-rate matcher (parametri zavise od TX konfiguracije)
        # TX mapira 960 QPSK simbola → 1920 bitova kroz 4 subfrejma.
        # Ovdje pretpostavljamo da RX skuplja kompletan blok (E_rx=1920).
        self.deratematcher = DeRateMatcher(E_rx=1920, N_coded=120)

        # Viterbi decoder (isti generatori kao u TX)
        self.viterbi = ViterbiDecoder(constraint_len=7,
                                      generators=[0o133, 0o171, 0o164],
                                      rate=1/3)

        # CRC checker
        self.crc = CRCChecker(poly=0x1021, init=0xFFFF)

    def process(self, rx_waveform: np.ndarray) -> dict:
        """
        Glavna metoda prijemnog lanca.
        Prima kompleksni signal iz kanala i vraća dict sa rezultatima.

        Parameters
        ----------
        rx_waveform : np.ndarray
            Kompleksni baznopojasni signal iz kanala.

        Returns
        -------
        dict
            Rezultati prijemnog lanca:
            - 'mib_bits': dekodirani MIB payload
            - 'crc_ok': bool flag CRC provjere
            - plus debug info iz svih koraka
        """

        debug = {}

        # 1) Validacija + RMS normalizacija
        rx_waveform = self.utils.validate_rx_samples(rx_waveform,
                                                     min_num_samples=128)
        rx_waveform, scale = self.utils.normalize_rms(rx_waveform,
                                                      target_rms=1.0)
        debug["rms_scale"] = scale

        # 2) PSS korelacija
        corr_metrics = self.pss_sync.correlate(rx_waveform)
        tau_hat, detected_nid = self.pss_sync.estimate_timing(corr_metrics)
        cfo_hat = self.pss_sync.estimate_cfo(rx_waveform, tau_hat, detected_nid)
        debug["pss_corr_metrics"] = corr_metrics
        debug["tau_hat"] = tau_hat
        debug["detected_nid"] = detected_nid
        debug["cfo_hat"] = cfo_hat

        # 3) CFO korekcija
        rx_corr = self.pss_sync.apply_cfo_correction(rx_waveform, cfo_hat)
        debug["rx_cfo_corrected"] = rx_corr

        # 4) OFDM demodulacija
        # Vrati i full FFT grid za debug, ali koristi ACTIVE grid dalje (72 nosioca za NDLRB=6).
        full_grid = self.ofdm_demod.demodulate(rx_corr, return_active_only=False)
        active_grid = self.ofdm_demod.demodulate(rx_corr, return_active_only=True)

        debug["grid_full_fft"] = full_grid
        debug["grid_active"] = active_grid

        # Brza validacija: očekujemo ndlrb * 12 aktivnih podnosača
        expected_active = self.ndlrb * 12
        assert active_grid.shape[1] == expected_active, (
            f"Active grid width mismatch: got {active_grid.shape[1]}, "
            f"expected {expected_active} for NDLRB={self.ndlrb}"
        )

        # 5) PBCH ekstrakcija (ISKLJUČIVO iz ACTIVE grida)
        pbch_symbols_rx = self.pbch_ext.extract(active_grid)
        debug["pbch_symbols_rx"] = pbch_symbols_rx

        # 6) QPSK demapiranje
        demapped_bits = self.demapper.demap(pbch_symbols_rx)
        debug["demapped_bits"] = demapped_bits

        # 7) De-rate-matching
        deratematched_bits = self.deratematcher.accumulate(demapped_bits, soft=False)
        debug["deratematched_bits"] = deratematched_bits

        # 8) Viterbi dekodiranje
        decoded_bits = self.viterbi.decode(deratematched_bits)
        debug["decoded_bits"] = decoded_bits

        # 9) CRC provjera
        payload_bits, crc_ok = self.crc.check(decoded_bits)
        debug["payload_bits"] = payload_bits
        debug["crc_ok"] = crc_ok

        return {
            "mib_bits": payload_bits,
            "crc_ok": crc_ok,
            "debug": debug
        }
