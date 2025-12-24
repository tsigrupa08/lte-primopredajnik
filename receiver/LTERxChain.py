import numpy as np

# Import svih modula 
from receiver.utils import RxUtils
from rx.pss_sync import PSSSynchronizer
from rx.ofdm_demodulator import OFDMDemodulator
from rx.resource_grid_extractor import PBCHExtractor
from rx.qpsk_demapper import QPSKDemapper
from rx.pbch_deratematch import DeRateMatcher
from rx.viterbi_decoder import ViterbiDecoder
from rx.crc_checker import CRCChecker


class LTERxChain:
    """
    LTE Receiver Chain (PBCH prijemnik).
    Spaja sve korake prijema u tačnom redoslijedu:
    1. Validacija i normalizacija signala
    2. PSS korelacija → detekcija N_ID_2, timing, CFO
    3. CFO korekcija
    4. OFDM demodulacija (CP remove + FFT)
    5. Ekstrakcija PBCH simbola iz grida
    6. QPSK demapiranje u bitove
    7. De-rate-matching (akumulacija)
    8. Viterbi dekodiranje
    9. CRC provjera → vraća payload (MIB) i CRC flag

    Rezultat je dict sa svim međukoracima (debug) i finalnim MIB bitovima.
    """
    """
Examples
--------
Kako se poziva i koristi LTERxChain klasa:

>>> import numpy as np
>>> from rx.LTERxChain import LTERxChain
>>>
>>> # Dummy signal (random kompleksni uzorci)
>>> rx_waveform = np.random.randn(4096) + 1j*np.random.randn(4096)
>>>
>>> # Inicijalizacija prijemnog lanca
>>> rx_chain = LTERxChain(sample_rate_hz=1.92e6, ndlrb=6, normal_cp=True)
>>>
>>> # Pokretanje prijema
>>> result = rx_chain.process(rx_waveform)
>>>
>>> # Rezultati
>>> print("CRC OK:", result["crc_ok"])
>>> print("Decoded MIB bits:", result["mib_bits"])
>>> print("Debug keys:", result["debug"].keys())

Notes
-----
- `mib_bits` : dekodirani MIB payload bitovi
- `crc_ok`   : bool flag CRC provjere
- `debug`    : dict sa svim međurezultatima (PSS metrike, grid, PBCH simboli,
               demapirani bitovi, deratematched bitovi, dekodirani bitovi itd.)
"""


    def __init__(self,
                 sample_rate_hz=1.92e6,
                 ndlrb=6,
                 normal_cp=True,
                 fft_size=None):
        # Utils za validaciju i RMS normalizaciju
        self.utils = RxUtils()

        # PSS sync
        self.pss_sync = PSSSynchronizer(sample_rate_hz=sample_rate_hz)

        # OFDM demodulator
        self.ofdm_demod = OFDMDemodulator(ndlrb=ndlrb,
                                          normal_cp=normal_cp,
                                          new_fft_size=fft_size)

        # PBCH extractor
        self.pbch_ext = PBCHExtractor()

        # QPSK demapper
        self.demapper = QPSKDemapper(mode="hard")

        # De-rate matcher (parametri zavise od tvoje TX konfiguracije)
        self.deratematcher = DeRateMatcher(E_rx=864, N_coded=192)

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
        cfo_hat = self.pss_sync.estimate_cfo(rx_waveform, tau_hat,
                                             detected_nid)
        debug["pss_corr_metrics"] = corr_metrics
        debug["tau_hat"] = tau_hat
        debug["detected_nid"] = detected_nid
        debug["cfo_hat"] = cfo_hat

        # 3) CFO korekcija
        rx_corr = self.pss_sync.apply_cfo_correction(rx_waveform, cfo_hat)
        debug["rx_cfo_corrected"] = rx_corr

        # 4) OFDM demodulacija
        grid = self.ofdm_demod.demodulate(rx_corr)
        debug["grid"] = grid

        # 5) PBCH ekstrakcija
        pbch_symbols_rx = self.pbch_ext.extract(grid)
        debug["pbch_symbols_rx"] = pbch_symbols_rx

        # 6) QPSK demapiranje
        demapped_bits = self.demapper.demap(pbch_symbols_rx)
        debug["demapped_bits"] = demapped_bits

        # 7) De-rate-matching
        deratematched_bits = self.deratematcher.accumulate(demapped_bits,
                                                           soft=False)
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
