import numpy as np

class OFDMModulator:
    """
    OFDM modulator za LTE signale.
    """

    def __init__(self, resource_grid: np.ndarray, new_fft_size: int | None = None):
        self.resource_grid: np.ndarray = resource_grid
        self.new_fft_size: int | None = new_fft_size
        self.num_subcarriers, self.num_ofdm_symbols = resource_grid.shape

        if self.num_subcarriers % 12 != 0:
            raise ValueError("Broj podnosioca mora biti višekratnik od 12.")
        
        self.ndlrb: int = self.num_subcarriers // 12
        self.normal_cp: bool = (self.num_ofdm_symbols % 14 == 0)

        if self.normal_cp:
            self.num_subframes: int = self.num_ofdm_symbols // 14
        else:
            self.num_subframes: int = self.num_ofdm_symbols // 12

        self.N: int = self._determine_fft_size()
        self.sample_rate: int = self.N * 15000

        multiplier = self.N // 128
        self.Ncp: np.ndarray = multiplier * np.array([10] + [9]*6)
        self.Ecp: np.ndarray = multiplier * np.array([32]*6)

        if self.normal_cp:
            self.n_symbols_per_slot: int = 7
            self.cp_lengths: np.ndarray = self.Ncp
            self.output_length: int = self.num_subframes * (self.N*14 + 2*self.Ncp[0] + 12*self.Ncp[1])
        else:
            self.n_symbols_per_slot: int = 6
            self.cp_lengths: np.ndarray = self.Ecp
            self.output_length: int = self.num_subframes * (self.N*12 + 12*self.Ecp[0])

    def _determine_fft_size(self) -> int:
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
        OFDM modulacija resource grid-a.

        Returns
        -------
        output_waveform : np.ndarray
            Time-domain OFDM signal
        sample_rate : int
            Sampling frequency
        """
        output_waveform: np.ndarray = np.zeros(self.output_length, dtype=complex)
        start_index: int = 0

        for sym_idx in range(self.num_ofdm_symbols):
            ifft_input: np.ndarray = np.zeros(self.N, dtype=complex)

            # DC bin
            dc_index = self.N // 2
            ifft_input[dc_index] = 0.0

            pos_freq_indices = np.arange(dc_index + 1, dc_index + 1 + self.num_subcarriers // 2)
            neg_freq_indices = np.arange(dc_index - self.num_subcarriers // 2, dc_index)

            pos_subcarriers = np.arange(self.num_subcarriers // 2, self.num_subcarriers)
            neg_subcarriers = np.arange(0, self.num_subcarriers // 2)

            ifft_input[pos_freq_indices] = self.resource_grid[pos_subcarriers, sym_idx]
            ifft_input[neg_freq_indices] = self.resource_grid[neg_subcarriers, sym_idx]

            ifft_output: np.ndarray = np.fft.ifft(ifft_input, self.N)

            sym_in_slot: int = sym_idx % self.n_symbols_per_slot
            cp_len: int = self.cp_lengths[sym_in_slot]
            cyclic_prefix: np.ndarray = ifft_output[-cp_len:]

            output_symbol: np.ndarray = np.concatenate([cyclic_prefix, ifft_output])

            end_index: int = start_index + len(output_symbol)
            output_waveform[start_index:end_index] = output_symbol
            start_index = end_index

        return output_waveform, self.sample_rate
