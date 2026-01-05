import unittest
import numpy as np

from transmitter.ofdm import OFDMModulator


class TestOFDMModulator(unittest.TestCase):
    """
    Unit testovi za LTE OFDMModulator:
    - NDLRB -> FFT mapping
    - CP integritet (normal + extended)
    - round-trip (OFDM -> FFT -> rekonstrukcija grida)
    - error handling
    """

    def setUp(self) -> None:
        self.ndlrb_list = [6, 15, 25, 50, 75, 100]

        # Ispravno LTE mapiranje FFT veličina (po standardnom LTE numerologiji)
        self.expected_fft_map = {
            6: 128,
            15: 256,
            25: 512,
            50: 1024,
            75: 1536,
            100: 2048,
        }

        self.rng = np.random.default_rng(42)

    def _random_grid(self, ndlrb: int, num_symbols: int) -> np.ndarray:
        """Slučajan kompleksni grid oblika (12*ndlrb, num_symbols)."""
        subcarriers = ndlrb * 12
        re = self.rng.standard_normal((subcarriers, num_symbols))
        im = self.rng.standard_normal((subcarriers, num_symbols))
        return (re + 1j * im).astype(np.complex128)

    # =================================================================
    # 1) OSNOVNE DIMENZIJE / PARAMETRI
    # =================================================================

    def test_01_sample_rate_correctness(self):
        """Sample rate mora biti N * 15000."""
        for ndlrb in self.ndlrb_list:
            grid = self._random_grid(ndlrb, 14)
            mod = OFDMModulator(grid)

            expected_sr = self.expected_fft_map[ndlrb] * 15000
            self.assertEqual(mod.sample_rate, expected_sr)

            mod_custom = OFDMModulator(grid, new_fft_size=2048)
            self.assertEqual(mod_custom.sample_rate, 2048 * 15000)

    def test_02_waveform_length_normal_cp(self):
        """Provjera dužine talasnog oblika za normal CP, 1 subframe (14 simbola)."""
        ndlrb = 25  # N=512
        grid = self._random_grid(ndlrb, 14)
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()

        self.assertEqual(len(wav), mod.output_length)

        # Za N=512 (normal CP): cp_first=40, cp_others=36
        # output_length = 14*N + 2*40 + 12*36 = 14*512 + 80 + 432 = 7680
        self.assertEqual(len(wav), 7680)

    # =================================================================
    # 2) CP INTEGRITET
    # =================================================================

    def test_03_cp_integrity_normal(self):
        """CP mora biti kopija zadnjih cp_len uzoraka simbola (normal CP)."""
        grid = self._random_grid(25, 14)
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()

        cp_len = int(mod.cp_lengths[0])
        N = int(mod.N)

        cp = wav[0:cp_len]
        data = wav[cp_len:cp_len + N]
        tail = data[-cp_len:]
        np.testing.assert_allclose(cp, tail, rtol=0.0, atol=1e-12)

    def test_04_cp_integrity_extended(self):
        """CP mora biti kopija zadnjih cp_len uzoraka simbola (extended CP)."""
        grid = self._random_grid(25, 12)  # extended: 12 simbola / subframe
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()

        cp_len = int(mod.cp_lengths[0])
        N = int(mod.N)
        sym_len = cp_len + N

        # uzmimo drugi simbol (index 1)
        start = sym_len
        cp = wav[start:start + cp_len]
        data = wav[start + cp_len:start + sym_len]
        tail = data[-cp_len:]
        np.testing.assert_allclose(cp, tail, rtol=0.0, atol=1e-12)

    # =================================================================
    # 3) DSP ROUND-TRIP (OFDM -> FFT -> REKONSTRUKCIJA GRID-a)
    # =================================================================

    def test_05_dsp_roundtrip_mse(self):
        """
        Round-trip: modulacija -> uklanjanje CP -> FFT -> ekstrakcija subcarriera.
        Mora rekonstruisati originalni grid (MSE veoma mali).

        Test pokriva obje TX varijante:
        - TX radi N*ifft(ifft_input)  -> FFT vraća centred spektar (DC u sredini)
        - TX radi N*ifft(ifftshift(ifft_input)) -> FFT vraća unshifted (DC na index 0),
          pa je potreban fftshift na RX strani.
        """
        ndlrb = 15
        grid_in = self._random_grid(ndlrb, 14)
        mod = OFDMModulator(grid_in)
        wav, _ = mod.modulate()

        N = int(mod.N)
        dc = N // 2
        half = mod.num_subcarriers // 2  # npr. 90 za ndlrb=15

        pos_sc = np.arange(half, 2 * half)   # drugi dio grida: pozitivne
        neg_sc = np.arange(0, half)          # prvi dio grida: negativne

        def reconstruct(scale_div: float, use_fftshift: bool) -> np.ndarray:
            rec = np.zeros_like(grid_in)
            cur = 0

            for i in range(mod.num_ofdm_symbols):
                cp = int(mod.cp_lengths[i % mod.n_symbols_per_slot])
                data_t = wav[cur + cp:cur + cp + N]

                data_f = np.fft.fft(data_t)
                if scale_div != 1.0:
                    data_f = data_f / scale_div

                if use_fftshift:
                    data_f = np.fft.fftshift(data_f)

                # ekstrakcija oko DC (DC u sredini)
                rec[pos_sc, i] = data_f[dc + 1: dc + 1 + half]
                rec[neg_sc, i] = data_f[dc - half: dc]

                cur += (cp + N)

            return rec

        # probaj sve kombinacije (skala i shift) i uzmi najbolju
        candidates = [
            reconstruct(scale_div=1.0, use_fftshift=False),
            reconstruct(scale_div=float(N), use_fftshift=False),
            reconstruct(scale_div=1.0, use_fftshift=True),
            reconstruct(scale_div=float(N), use_fftshift=True),
        ]
        mses = [np.mean(np.abs(grid_in - r) ** 2) for r in candidates]
        mse = float(min(mses))

        self.assertLess(mse, 1e-18)

    # =================================================================
    # 4) SPECIJALNI SLUČAJEVI
    # =================================================================

    def test_06_zero_input_response(self):
        """Ako je grid nula, izlaz mora biti nula."""
        grid = np.zeros((25 * 12, 14), dtype=np.complex128)
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()
        self.assertTrue(np.allclose(wav, 0.0))

    def test_07_single_active_subcarrier_constant_envelope(self):
        """
        Jedan aktivan podnosioc -> vremenski signal je kompleksni eksponencijal
        (magnituda približno konstantna).
        """
        grid = np.zeros((25 * 12, 14), dtype=np.complex128)
        grid[0, 0] = 1.0 + 0j
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()

        first_symbol = wav[: int(mod.cp_lengths[0]) + int(mod.N)]
        mags = np.abs(first_symbol)

        # treba biti praktično konstantno
        self.assertLess(np.std(mags), 1e-4)

    def test_08_oversampling_logic(self):
        """new_fft_size mora podići N i proporcionalno skalirati CP."""
        grid = self._random_grid(6, 14)
        target_fft = 512
        mod = OFDMModulator(grid, new_fft_size=target_fft)

        self.assertEqual(mod.N, target_fft)

        expected_cp0 = int(160 * target_fft / 2048)  # skala normal CP (prvi simbol)
        self.assertEqual(int(mod.cp_lengths[0]), expected_cp0)

    # =================================================================
    # 5) ERROR HANDLING
    # =================================================================

    def test_09_error_invalid_subcarrier_count(self):
        """Greška ako broj redova nije djeljiv sa 12."""
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((13, 14), dtype=np.complex128))

    def test_10_error_invalid_symbol_count(self):
        """Greška ako broj simbola nije višekratnik 14 (normal) ili 12 (extended)."""
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((72, 5), dtype=np.complex128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
