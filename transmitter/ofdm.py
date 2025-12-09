import numpy as np

class OFDMModulator:
    """
    Klasa za generisanje LTE OFDM signala iz resource grida.

    Ova klasa implementira IFFT, dodavanje cikličkog prefiksa (CP) i
    generisanje vremenskog talasnog oblika za LTE OFDM transmisiju.
    """

    def __init__(self, resource_grid, new_fft_size=None):
        """
        Inicijalizacija OFDM modulatora.

        Parameters
        ----------
        resource_grid : np.ndarray
            Matrica dimenzija (NDLRB * 12, broj OFDM simbola) koja sadrži
            frekvencijski raspoređene QAM simbole.
        new_fft_size : int, optional
            Override za veličinu FFT-a. Mora biti jedna od [128, 256, 512, 1024, 1408, 1536, 2048].

        Raises
        ------
        ValueError
            Ako broj podnosioca nije višekratnik od 12 ili broj simbola nije validan.
        """
        self.resource_grid = resource_grid
        self.new_fft_size = new_fft_size
        self.num_subcarriers, self.num_ofdm_symbols = resource_grid.shape

        if self.num_subcarriers % 12 != 0:
            raise ValueError("Broj podnosioca mora biti višekratnik od 12.")
        
        self.ndlrb = self.num_subcarriers // 12
        self.normal_cp = (self.num_ofdm_symbols % 14 == 0)
        self.extended_cp = (self.num_ofdm_symbols % 12 == 0)

        if not (self.normal_cp or self.extended_cp):
            raise ValueError("Broj OFDM simbola mora biti višekratnik od 12 ili 14.")

        if self.normal_cp:
            self.num_subframes = self.num_ofdm_symbols // 14
        else:
            self.num_subframes = self.num_ofdm_symbols // 12

        self.N = self._determine_fft_size()
        self.sample_rate = self.N * 15000  # LTE spacing

        multiplier = self.N // 128
        self.Ncp = multiplier * np.array([10] + [9]*6)
        self.Ecp = multiplier * np.array([32]*6)

        if self.normal_cp:
            self.n_symbols_per_slot = 7
            self.cp_lengths = self.Ncp
            self.output_length = self.num_subframes * (self.N*14 + 2*self.Ncp[0] + 12*self.Ncp[1])
        else:
            self.n_symbols_per_slot = 6
            self.cp_lengths = self.Ecp
            self.output_length = self.num_subframes * (self.N*12 + 12*self.Ecp[0])

    def _determine_fft_size(self):
        """
        Određuje minimalnu dozvoljenu FFT veličinu na osnovu broja resource blokova.

        Returns
        -------
        int
            Veličina FFT-a koja se koristi za IFFT.

        Raises
        ------
        ValueError
            Ako NDLRB nije podržan ili ako je override FFT veličina nevalidna.
        """
        fft_map = {
            6: 128,
            15: 256,
            25: 512,
            50: 1024,
            75: 1408,
            100: 1408
        }
        if self.ndlrb not in fft_map:
            raise ValueError("NDLRB mora biti jedan od [6, 15, 25, 50, 75, 100].")
        
        base_fft = fft_map[self.ndlrb]
        if self.new_fft_size is None:
            return base_fft
        else:
            valid_sizes = [128, 256, 512, 1024, 1408, 1536, 2048]
            if self.new_fft_size not in valid_sizes:
                raise ValueError("Nevalidna FFT veličina.")
            return max(base_fft, self.new_fft_size)

    def modulate(self):
        """
        Generiše vremenski OFDM signal sa cikličkim prefiksom.

        Za svaki OFDM simbol:
        - mapira frekvencijske komponente u IFFT ulaz
        - izvršava IFFT
        - dodaje ciklički prefiks
        - spaja sve simbole u jedan vremenski signal

        Returns
        -------
        tuple
            output_waveform : np.ndarray
                Kompletnan vremenski signal sa CP.
            sample_rate : int
                Sample rate signala u Hz.

        Raises
        ------
        RuntimeError
            Ako izlazni signal nije potpuno popunjen.
        """
        output_waveform = np.zeros(self.output_length, dtype=complex)

        pos_subcarriers = np.arange(self.num_subcarriers//2, self.num_subcarriers)
        neg_subcarriers = np.arange(0, self.num_subcarriers//2)

        pos_freq_indices = np.arange(0, self.num_subcarriers//2)
        neg_freq_indices = np.arange(self.N - self.num_subcarriers//2, self.N)

        start_index = 0
        for sym_idx in range(self.num_ofdm_symbols):
            ifft_input = np.zeros(self.N, dtype=complex)
            ifft_input[pos_freq_indices] = self.resource_grid[pos_subcarriers, sym_idx]
            ifft_input[neg_freq_indices] = self.resource_grid[neg_subcarriers, sym_idx]

            ifft_output = self.N * np.fft.ifft(ifft_input, self.N)

            sym_in_slot = sym_idx % self.n_symbols_per_slot
            cp_len = self.cp_lengths[sym_in_slot]
            cyclic_prefix = ifft_output[-cp_len:]

            output_symbol = np.concatenate([cyclic_prefix, ifft_output])

            end_index = start_index + len(output_symbol)
            output_waveform[start_index:end_index] = output_symbol
            start_index = end_index

        if start_index != self.output_length:
            raise RuntimeError("Output waveform nije potpuno popunjen.")

        return output_waveform, self.sample_rate