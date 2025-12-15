import unittest
import numpy as np

from transmitter.ofdm import OFDMModulator


class TestOFDMModulator(unittest.TestCase):
    """
    Unit testovi za LTE OFDMModulator (NDLRB mapiranje, CP, FFT/CP dimenzije, error handling).
    """

    def setUp(self) -> None:
        self.ndlrb_list = [6, 15, 25, 50, 75, 100]
        self.base_fft_map = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 1408, 100: 1408}

        # fiksni RNG radi ponovljivosti
        self.rng = np.random.default_rng(42)

    def _random_grid(self, ndlrb: int, num_symbols: int) -> np.ndarray:
        """Generiše slučajan kompleksni grid oblika (12*ndlrb, num_symbols)."""
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
            expected_sr = self.base_fft_map[ndlrb] * 15000
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
        self.assertEqual(len(wav), 7680)  # očekivano za N=512 i normal CP

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
        grid = self._random_grid(25, 12)  # extended (12 simbola / subframe)
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

        Napomena: test je robustan i za varijantu sa ifft ili N*ifft,
        jer automatski bira bolju skalu.
        """
        ndlrb = 15
        grid_in = self._random_grid(ndlrb, 14)
        mod = OFDMModulator(grid_in)
        wav, _ = mod.modulate()

        N = int(mod.N)
        dc = N // 2
        half = mod.num_subcarriers // 2

        pos_sc = np.arange(mod.num_subcarriers // 2, mod.num_subcarriers)
        neg_sc = np.arange(0, mod.num_subcarriers // 2)

        def reconstruct(scale_div: float) -> np.ndarray:
            rec = np.zeros_like(grid_in)
            cur = 0
            for i in range(mod.num_ofdm_symbols):
                cp = int(mod.cp_lengths[i % mod.n_symbols_per_slot])
                data_t = wav[cur + cp:cur + cp + N]
                data_f = np.fft.fft(data_t)

                if scale_div != 1.0:
                    data_f = data_f / scale_div

                # ekstrakcija tačno kao u mapperu oko DC
                rec[pos_sc, i] = data_f[dc + 1: dc + 1 + half]
                rec[neg_sc, i] = data_f[dc - half: dc]

                cur += (cp + N)
            return rec

        # probaj obje varijante skale (za ifft i za N*ifft)
        rec_a = reconstruct(scale_div=1.0)
        rec_b = reconstruct(scale_div=float(N))

        mse_a = np.mean(np.abs(grid_in - rec_a) ** 2)
        mse_b = np.mean(np.abs(grid_in - rec_b) ** 2)
        mse = min(mse_a, mse_b)

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
        (magnituda približno konstantna). Testiramo varijaciju magnitude.
        """
        grid = np.zeros((25 * 12, 14), dtype=np.complex128)
        grid[0, 0] = 1.0 + 0j
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()

        first_symbol = wav[: int(mod.cp_lengths[0]) + int(mod.N)]
        mags = np.abs(first_symbol)
        self.assertLess(np.std(mags), 1e-4)

    def test_08_oversampling_logic(self):
        """new_fft_size mora podići N i proporcionalno skalirati CP."""
        grid = self._random_grid(6, 14)
        target_fft = 512
        mod = OFDMModulator(grid, new_fft_size=target_fft)

        self.assertEqual(mod.N, target_fft)

        expected_cp0 = int(160 * target_fft / 2048)  # LTE skala za normal CP (prvi simbol)
        self.assertEqual(int(mod.cp_lengths[0]), expected_cp0)

    # =================================================================
    # 5) ERROR HANDLING
    # =================================================================

    def test_09_error_invalid_subcarrier_count(self):
        """Greška ako broj redova nije djeljiv sa 12."""
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((13, 14), dtype=np.complex128))

    def test_10_error_invalid_symbol_count(self):
        """
        Greška ako broj simbola nije višekratnik 14 (normal) ili 12 (extended).

        Ako ti ovaj test padne: znači da u OFDMModulator.__init__ treba dodati
        validaciju (kao što si ranije imala u staroj verziji).
        """
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((72, 5), dtype=np.complex128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
