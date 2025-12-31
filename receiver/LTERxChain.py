import numpy as np

# --- IMPORTI ---
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
    LTE Receiver Chain za PBCH (Physical Broadcast Channel).

    Ova klasa integrira sve komponente prijemnika potrebne za demodulaciju i
    dekodiranje Master Information Block-a (MIB) iz primljenog LTE signala.
    
    Implementira robustan mehanizam obrade koji uključuje padding signala
    i hvatanje izuzetaka (try-except) kako bi se osigurala stabilnost
    simulacije čak i pri vrlo niskim SNR vrijednostima kada sinkronizacija
    može biti neprecizna.

    Namjena
    -------
    Koristi se u 'End-to-End' simulacijama za evaluaciju performansi (BER/BLER)
    LTE fizičkog sloja. Klasa apstrahuje kompleksnost pojedinačnih blokova
    (sinkronizacija, demodulacija, dekodiranje) u jedan jednostavan interfejs.

    Koraci obrade
    -------------
    1. **Validacija i normalizacija:** Provjera ulaznog signala i skaliranje snage.
    2. **PSS Sinkronizacija:** Pronalazak početka okvira (timing) i detekcija Cell ID-a.
    3. **CFO Procjena i Korekcija:** Uklanjanje frekvencijskog pomaka.
    4. **Timing Poravnanje:** Premotavanje indeksa na početak subfrejma uz sigurnosni padding.
    5. **OFDM Demodulacija:** FFT transformacija i uklanjanje CP-a.
    6. **PBCH Ekstrakcija:** Izdvajanje QPSK simbola iz resursnog grida.
    7. **Dekodiranje:** QPSK demapiranje, De-Rate matching, Viterbi dekodiranje i CRC provjera.

    Parametri
    ---------
    sample_rate_hz : float, optional
        Frekvencija uzorkovanja u Hz. Default je 1.92e6 (za 1.4 MHz LTE).
    ndlrb : int, optional
        Broj downlink resursnih blokova. Default je 6.
    normal_cp : bool, optional
        True za normalni ciklički prefiks, False za prošireni. Default je True.
    fft_size : int, optional
        Eksplicitna veličina FFT-a. Ako je None, određuje se na osnovu ndlrb.

    Atributi
    --------
    pss_sync : PSSSynchronizer
        Objekat za vremensku i frekvencijsku sinkronizaciju.
    ofdm_demod : OFDMDemodulator
        Objekat za konverziju vremenskog signala u frekvencijski grid.
    pbch_ext : PBCHExtractor
        Objekat za izdvajanje PBCH resursnih elemenata.
    viterbi : ViterbiDecoder
        Objekat za konvolucijsko dekodiranje (FEC).

    Primjeri
    --------
    >>> import numpy as np
    >>> from receiver.LTERxChain import LTERxChain
    >>> 
    >>> # 1. Inicijalizacija prijemnika
    >>> rx = LTERxChain(sample_rate_hz=1.92e6, ndlrb=6)
    >>> 
    >>> # 2. Generisanje dummy signala (ili učitavanje iz fajla)
    >>> # Ovdje samo pravimo šumni signal kao placeholder
    >>> dummy_waveform = (np.random.randn(3840) + 1j*np.random.randn(3840)).astype(np.complex64)
    >>> 
    >>> # 3. Procesiranje
    >>> result = rx.process(dummy_waveform)
    >>> 
    >>> # 4. Provjera rezultata
    >>> if result['crc_ok']:
    ...     print(f"Uspješno dekodiran MIB: {result['mib_bits']}")
    ... else:
    ...     print("CRC provjera nije prošla (signal previše šuman ili nema PBCH).")
    """

    def __init__(
        self,
        sample_rate_hz: float = 1.92e6,
        ndlrb: int = 6,
        normal_cp: bool = True,
        fft_size: int | None = None
    ):
        self.utils = RxUtils()
        self.sample_rate_hz = float(sample_rate_hz)
        self.ndlrb = int(ndlrb)
        self.normal_cp = bool(normal_cp)

        # 1. Komponente za sinkronizaciju i demodulaciju
        self.pss_sync = PSSSynchronizer(sample_rate_hz=self.sample_rate_hz)
        self.ofdm_demod = OFDMDemodulator(
            ndlrb=self.ndlrb,
            normal_cp=self.normal_cp,
            new_fft_size=fft_size
        )

        # 2. Konfiguracija PBCH Ekstraktora
        # Očekujemo 4 subfrejma da bismo skupili 960 simbola (4 * 240)
        num_subframes_fixed = 4
        symbols_per_sf = 14 if self.normal_cp else 12
        
        pbch_indices = []
        for sf in range(num_subframes_fixed):
            base = sf * symbols_per_sf
            # PBCH je u simbolima 7, 8, 9, 10 svakog subfrejma
            pbch_indices.extend([base + 7, base + 8, base + 9, base + 10])

        pbch_cfg = PBCHConfig(
            ndlrb=self.ndlrb,
            normal_cp=self.normal_cp,
            pbch_symbol_indices=pbch_indices,
            pbch_symbols_per_subframe=960
        )
        self.pbch_ext = PBCHExtractor(pbch_cfg)

        # 3. Dekodiranje
        self.demapper = QPSKDemapper(mode="hard")
        self.deratematcher = DeRateMatcher(E_rx=1920, N_coded=120)
        self.viterbi = ViterbiDecoder(
            constraint_len=7,
            generators=[0o133, 0o171, 0o164],
            rate=1/3
        )
        self.crc = CRCChecker(poly=0x1021, init=0xFFFF)

    def process(self, rx_waveform: np.ndarray) -> dict:
        """
        Obrađuje ulazni vremenski signal i pokušava dekodirati MIB.

        Ova metoda izvodi kompletan prijemni lanac. Uključuje 'try-except' blok
        koji hvata greške nastale uslijed lošeg kvaliteta signala (npr. kada
        PSS sinkronizacija promaši, pa OFDM demodulator nema dovoljno uzoraka).
        U takvim slučajevima, metoda "gracefully" vraća neuspjeh (CRC=False)
        umjesto da sruši program.

        Parametri
        ---------
        rx_waveform : np.ndarray
            Kompleksni bazni signal (IQ uzorci).

        Povratna vrijednost
        -------------------
        dict
            Rječnik sa sljedećim ključevima:
            - 'mib_bits' : np.ndarray
                Dekodirani informacioni bitovi (24 bita). Ako dekodiranje ne uspije,
                vraća niz nula.
            - 'crc_ok' : bool
                True ako je CRC provjera uspješna, False inače.
            - 'debug' : dict
                Dodatne informacije o procesu (procjenjeni timing, CFO, grid).

        Raises
        ------
        Nijedan. Sve interne greške (ValueError, IndexError) se hvataju i 
        rezultiraju povratkom `crc_ok=False`.
        """
        debug = {}

        try:
            # 1. Validacija
            rx_waveform = self.utils.validate_rx_samples(rx_waveform, min_num_samples=128)
            rx_waveform, scale = self.utils.normalize_rms(rx_waveform, target_rms=1.0)
            debug["rms_scale"] = scale

            # 2. PSS Sinkronizacija
            corr = self.pss_sync.correlate(rx_waveform)
            tau_hat, detected_nid = self.pss_sync.estimate_timing(corr)
            cfo_hat = self.pss_sync.estimate_cfo(rx_waveform, tau_hat, detected_nid)

            debug["tau_hat"] = tau_hat
            debug["cfo_hat"] = cfo_hat

            # 3. CFO Korekcija
            rx_corr = self.pss_sync.apply_cfo_correction(rx_waveform, cfo_hat)

            # 4. Timing Korekcija (Premotavanje na početak frejma)
            # PSS je na simbolu #6. Moramo se vratiti na simbol #0.
            symbols_before_pss = 6 if self.normal_cp else 5
            
            samples_offset = 0
            for i in range(symbols_before_pss):
                cp_len = self.ofdm_demod.cp_lengths[i % 7]
                samples_offset += (self.ofdm_demod.fft_size + cp_len)
                
            frame_start_index = tau_hat - samples_offset

            # --- FIX: PADDING I REZANJE ---
            
            # A) Ako je početak prije nule (zbog šuma), dodaj padding naprijed
            if frame_start_index < 0:
                padding_front = np.zeros(abs(frame_start_index), dtype=rx_corr.dtype)
                rx_synchronized = np.concatenate([padding_front, rx_corr])
            else:
                rx_synchronized = rx_corr[frame_start_index:]

            # B) Safety Padding NA KRAJU (Ključno za sprečavanje "IndexError")
            # Dodajemo cca 3ms praznog signala na kraj. 
            # Ovo osigurava da demodulator uvijek ima dovoljno uzoraka, 
            # čak i ako je sinkronizacija malo promašila.
            safety_pad_len = int(self.sample_rate_hz * 0.003) 
            padding_back = np.zeros(safety_pad_len, dtype=rx_synchronized.dtype)
            rx_synchronized = np.concatenate([rx_synchronized, padding_back])

            # 5. OFDM Demodulacija
            grid_fft = self.ofdm_demod.demodulate(rx_synchronized)

            # 6. Ekstrakcija aktivnih sub carriera
            active = self.ofdm_demod.extract_active_subcarriers(grid_fft)
            grid_pbch = active.T
            debug["grid_active"] = grid_pbch

            # 7. Dekodiranje
            # Pokušavamo izvući podatke. Ako je grid ipak preloš/prekratak, 
            # extract() metoda bi mogla baciti ValueError.
            pbch_symbols_rx = self.pbch_ext.extract(grid_pbch)
            
            demapped_bits = self.demapper.demap(pbch_symbols_rx)
            deratematched_bits = self.deratematcher.accumulate(demapped_bits, soft=False)
            decoded_bits = self.viterbi.decode(deratematched_bits)
            payload_bits, crc_ok = self.crc.check(decoded_bits)

            return {
                "mib_bits": payload_bits,
                "crc_ok": crc_ok,
                "debug": debug
            }

        except ValueError:
            # Ako se desi bilo kakva greška tokom procesiranja (npr. grid prekratak),
            # to tretiramo kao neuspješan prijem (CRC Fail).
            # Ovo omogućava simulaciji da nastavi dalje.
            return {
                "mib_bits": np.zeros(24, dtype=int), # Vraćamo prazne bitove
                "crc_ok": False,                     # Prijem nije uspio
                "debug": debug
            }