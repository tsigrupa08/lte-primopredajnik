import numpy as np


class OFDMModulator:
    """
    OFDM modulator za LTE signale (downlink, baseband).

    Ova klasa generiše vremenski OFDM talasni oblik iz LTE resource grida:
    - mapira centralnih ``12 * NDLRB`` podnosioca u IFFT ulaz,
    - postavlja DC bin na nulu,
    - računa IFFT (skalirano sa ``N``),
    - dodaje ciklički prefiks (CP) za svaki OFDM simbol,
    - spaja sve simbole u jedan vremenski signal.

    Parametri
    ----------
    resource_grid : np.ndarray
        Kompleksna 2D matrica oblika ``(12*NDLRB, Nsym)`` gdje:
        - redovi predstavljaju podnosioca (subcarrier index u gridu),
        - kolone predstavljaju OFDM simbole.
    new_fft_size : int, opcionalno
        Ručni override FFT veličine. Ako je zadano, mora biti u listi
        ``[128, 256, 512, 1024, 1408, 1536, 2048]`` i biće uzeto
        ``max(base_fft, new_fft_size)``.

    Atributi
    --------
    ndlrb : int
        Broj downlink resource blokova (NDLRB), izveden iz broja podnosioca.
    normal_cp : bool
        True ako je broj simbola djeljiv sa 14 (normal CP).
    extended_cp : bool
        True ako je broj simbola djeljiv sa 12 (extended CP).
    N : int
        FFT veličina koja se koristi za IFFT.
    sample_rate : int
        Sampling frekvencija u Hz (``N * 15000``).
    output_length : int
        Ukupan broj uzoraka vremenskog signala nakon dodavanja CP.

    Raises
    ------
    ValueError
        Ako `resource_grid` nije 2D, ako broj podnosioca nije višekratnik 12,
        ili ako broj OFDM simbola nije kompatibilan sa LTE (14 ili 12 po subfrejmu).
    RuntimeError
        Ako dođe do nekonzistentnosti u popunjavanju izlaznog signala.

    Primjer
    -------
    Minimalni primjer (NDLRB=6, 1 subframe, normal CP)::

        >>> import numpy as np
        >>> from transmitter.ofdm import OFDMModulator
        >>>
        >>> # Prazan grid (72 podnosioca, 14 simbola)
        >>> grid = np.zeros((72, 14), dtype=np.complex64)
        >>>
        >>> # Npr. ubaci jedan simbol na neki podnosioc/simbol
        >>> grid[10, 0] = 1 + 1j
        >>>
        >>> ofdm = OFDMModulator(grid)
        >>> waveform, fs = ofdm.modulate()
        >>> waveform.shape
        (2048,)
        >>> fs
        1920000
    """

    def __init__(self, resource_grid: np.ndarray, new_fft_size: int | None = None) -> None:
        self.resource_grid: np.ndarray = np.asarray(resource_grid)
        self.new_fft_size: int | None = new_fft_size

        if self.resource_grid.ndim != 2:
            raise ValueError("resource_grid mora biti 2D matrica oblika (subcarriers, ofdm_symbols).")

        self.num_subcarriers, self.num_ofdm_symbols = self.resource_grid.shape

        if self.num_subcarriers % 12 != 0:
            raise ValueError("Broj podnosioca mora biti višekratnik od 12 (12*NDLRB).")

        self.ndlrb: int = self.num_subcarriers // 12

        # CP detekcija
        self.normal_cp: bool = (self.num_ofdm_symbols % 14 == 0)
        self.extended_cp: bool = (self.num_ofdm_symbols % 12 == 0)

        if not (self.normal_cp or self.extended_cp):
            raise ValueError(
                "Broj OFDM simbola mora biti višekratnik od 14 (normal CP) ili 12 (extended CP)."
            )

        if self.normal_cp:
            self.num_subframes: int = self.num_ofdm_symbols // 14
        else:
            self.num_subframes: int = self.num_ofdm_symbols // 12

        # FFT veličina + sample rate
        self.N: int = self._determine_fft_size()
        self.sample_rate: int = self.N * 15000  # LTE spacing: 15 kHz

        # CP dužine (skalirane prema N=128 bazi)
        multiplier = self.N // 128
        self.Ncp: np.ndarray = multiplier * np.array([10] + [9] * 6)   # normal CP (7 simbola/slot)
        self.Ecp: np.ndarray = multiplier * np.array([32] * 6)         # extended CP (6 simbola/slot)

        if self.normal_cp:
            self.n_symbols_per_slot: int = 7
            self.cp_lengths: np.ndarray = self.Ncp
            # ukupno CP po subfrejmu = 2*(10 + 6*9) = 128 (za N=128 bazu)
            self.output_length: int = self.num_subframes * (self.N * 14 + 2 * self.Ncp[0] + 12 * self.Ncp[1])
        else:
            self.n_symbols_per_slot: int = 6
            self.cp_lengths: np.ndarray = self.Ecp
            self.output_length: int = self.num_subframes * (self.N * 12 + 12 * self.Ecp[0])

    def _determine_fft_size(self) -> int:
        """
        Određuje minimalnu FFT veličinu prema NDLRB.

        Returns
        -------
        int
            FFT veličina (N) koja se koristi u IFFT.

        Raises
        ------
        ValueError
            Ako NDLRB nije podržan ili je `new_fft_size` nevalidan.
        """
        fft_map = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 1408, 100: 1408}
        if self.ndlrb not in fft_map:
            raise ValueError("NDLRB mora biti jedan od [6, 15, 25, 50, 75, 100].")

        base_fft = fft_map[self.ndlrb]

        if self.new_fft_size is None:
            return base_fft

        valid_sizes = [128, 256, 512, 1024, 1408, 1536, 2048]
        if self.new_fft_size not in valid_sizes:
            raise ValueError("Nevalidna FFT veličina.")
        return max(base_fft, self.new_fft_size)

    def modulate(self) -> tuple[np.ndarray, int]:
        """
        Izvršava OFDM modulaciju nad resource gridom.

        Za svaki OFDM simbol:
        1) kreira IFFT ulaz dužine `N`,
        2) mapira podnosioca oko DC bina (`N/2`) i ostavlja DC na nuli,
        3) računa IFFT (skalirano: ``N * ifft``),
        4) dodaje CP odgovarajuće dužine,
        5) konkatenira u izlazni signal.

        Returns
        -------
        output_waveform : np.ndarray
            Kompleksni vremenski signal sa CP (1D niz).
        sample_rate : int
            Sampling frekvencija u Hz.

        Raises
        ------
        RuntimeError
            Ako dođe do prelaska preko `output_length` ili ako izlaz nije
            potpuno popunjen.

        Primjer
        -------
        ::

            >>> import numpy as np
            >>> from transmitter.ofdm import OFDMModulator
            >>> grid = np.zeros((72, 14), dtype=np.complex64)
            >>> grid[36, 0] = 1 + 0j
            >>> ofdm = OFDMModulator(grid)
            >>> y, fs = ofdm.modulate()
            >>> y.shape[0] == ofdm.output_length
            True
        """
        output_waveform: np.ndarray = np.zeros(self.output_length, dtype=complex)
        start_index: int = 0

        dc_index = self.N // 2  # DC bin u sredini FFT-a

        # indeksi u IFFT ulazu (oko DC)
        pos_freq_indices = np.arange(dc_index + 1, dc_index + 1 + self.num_subcarriers // 2)
        neg_freq_indices = np.arange(dc_index - self.num_subcarriers // 2, dc_index)

        # redovi u gridu (npr. 72 za NDLRB=6)
        pos_subcarriers = np.arange(self.num_subcarriers // 2, self.num_subcarriers)
        neg_subcarriers = np.arange(0, self.num_subcarriers // 2)

        for sym_idx in range(self.num_ofdm_symbols):
            ifft_input: np.ndarray = np.zeros(self.N, dtype=complex)

            # DC bin je uvijek nula (ne koristi se)
            ifft_input[dc_index] = 0.0

            # mapiranje grida u IFFT bins
            ifft_input[pos_freq_indices] = self.resource_grid[pos_subcarriers, sym_idx]
            ifft_input[neg_freq_indices] = self.resource_grid[neg_subcarriers, sym_idx]

            # IFFT (skalirano sa N)
            ifft_output: np.ndarray = self.N * np.fft.ifft(ifft_input, self.N)

            sym_in_slot: int = sym_idx % self.n_symbols_per_slot
            cp_len: int = int(self.cp_lengths[sym_in_slot])
            cyclic_prefix: np.ndarray = ifft_output[-cp_len:]

            output_symbol: np.ndarray = np.concatenate([cyclic_prefix, ifft_output])

            end_index: int = start_index + output_symbol.size
            if end_index > self.output_length:
                raise RuntimeError("Prelazak preko output_length — provjeri CP dužine i broj simbola.")

            output_waveform[start_index:end_index] = output_symbol
            start_index = end_index

        if start_index != self.output_length:
            raise RuntimeError("Output waveform nije potpuno popunjen (mismatch u output_length kalkulaciji).")

        return output_waveform, self.sample_rate
