import numpy as np


class OFDMDemodulator:
    """
    OFDM Demodulator (CP removal + FFT)

    Ova klasa prima vremenski OFDM signal (rx_waveform),
    uklanja ciklički prefiks (CP) i radi FFT po OFDM simbolima,
    čime dobiva frekvencijski OFDM grid.

    Parametri su usklađeni s OFDMModulatorom (LTE-like postavka).
    """

    def __init__(self, ndlrb=6, normal_cp=True, new_fft_size=None):
        """
        Parametri
        ----------
        ndlrb : int
            Broj downlink resource blockova (npr. 6 → 72 subcarriera)
        normal_cp : bool
            True → normal cyclic prefix (LTE normal CP)
        new_fft_size : int ili None
            Ako je zadano, ručno postavlja FFT size
        """

        self.ndlrb = ndlrb
        self.normal_cp = normal_cp

        # FFT size (ako nije ručno zadan, određuje se iz LTE mape)
        self.fft_size = (
            new_fft_size
            if new_fft_size is not None
            else self._determine_fft_size(ndlrb)
        )

        # Broj aktivnih subcarrier-a
        self.n_subcarriers = 12 * ndlrb

        # Duljine cikličkog prefiksa po OFDM simbolu
        self.cp_lengths = self._determine_cp_lengths()

        # Sample rate (15 kHz spacing × FFT size)
        self.sample_rate = 15_000 * self.fft_size

    # ==============================================================
    # PARAMETRI OFDM-a (idealno za zajednički OFDMParams)
    # ==============================================================

    @staticmethod
    def _determine_fft_size(ndlrb):
        """
        Određuje FFT size prema LTE standardnoj mapi.
        """
        fft_map = {
            6: 128,
            15: 256,
            25: 512,
            50: 1024,
            75: 1536,
            100: 2048,
        }

        if ndlrb not in fft_map:
            raise ValueError(f"Nepodržan ndlrb = {ndlrb}")

        return fft_map[ndlrb]

    def _determine_cp_lengths(self):
        """
        Vraća listu duljina CP-a po OFDM simbolu.

        LTE normal CP:
        - prvi simbol u slotu ima duži CP
        - ostali simboli imaju kraći CP
        """

        if not self.normal_cp:
            raise NotImplementedError("Extended CP nije implementiran")

        # Referentne LTE vrijednosti (za FFT=2048)
        cp_first_ref = 160
        cp_others_ref = 144

        # Skaliranje CP duljine prema FFT size
        cp_first = int(self.fft_size * cp_first_ref / 2048)
        cp_others = int(self.fft_size * cp_others_ref / 2048)

        # 7 OFDM simbola po slotu
        return [cp_first] + [cp_others] * 6

    # ==============================================================
    # OFDM DEMODULACIJA
    # ==============================================================

    def demodulate(self, rx_waveform):
        """
        Demodulira OFDM signal iz vremenske u frekvencijsku domenu.

        Parametri
        ----------
        rx_waveform : numpy.ndarray (complex)
            Vremenski OFDM signal s cikličkim prefiksom

        Povrat
        -------
        grid : numpy.ndarray (complex)
            OFDM grid dimenzija:
            [broj_OFDM_simbola, fft_size]
        """

        # Osiguraj numpy kompleksni tip
        rx_waveform = np.asarray(rx_waveform, dtype=np.complex64)

        symbols_freq = []     # lista OFDM simbola u frekvenciji
        sample_idx = 0        # indeks unutar rx_waveform
        symbol_idx = 0        # redni broj OFDM simbola

        # Petlja dok god ima dovoljno uzoraka za cijeli OFDM simbol
        while True:
            cp_len = self.cp_lengths[symbol_idx % 7]
            total_len = cp_len + self.fft_size

            if sample_idx + total_len > len(rx_waveform):
                break

            # --------------------------------------------------
            # 1. Uklanjanje cikličkog prefiksa (CP removal)
            # --------------------------------------------------
            ofdm_symbol_time = rx_waveform[
                sample_idx + cp_len : sample_idx + total_len
            ]

            # --------------------------------------------------
            # 2. FFT (vremenska → frekvencijska domena)
            # --------------------------------------------------
            ofdm_symbol_freq = np.fft.fft(ofdm_symbol_time)

            # --------------------------------------------------
            # 3. FFT shift (DC subcarrier u sredinu spektra)
            # --------------------------------------------------
            ofdm_symbol_freq = np.fft.fftshift(ofdm_symbol_freq)

            symbols_freq.append(ofdm_symbol_freq)

            # Pomakni se na sljedeći OFDM simbol
            sample_idx += total_len
            symbol_idx += 1

        if len(symbols_freq) == 0:
            raise ValueError("Ulazni signal je prekratak za OFDM demodulaciju")

        # Stack → OFDM grid
        grid = np.vstack(symbols_freq)

        return grid

    # ==============================================================
    # POMOĆNA FUNKCIJA (često korisna kasnije)
    # ==============================================================

    def extract_active_subcarriers(self, grid):
        """
        Izdvaja samo aktivne subcarriere iz OFDM grida
        (korisno za channel estimation / demapping).
        """

        center = self.fft_size // 2
        half = self.n_subcarriers // 2

        return grid[:, center - half : center + half]
