import unittest
import numpy as np
import sys
import os

"""
Modul za testiranje OFDM Modulatora.
Verzija sa 10 testova (uklonjeni testovi za Extended CP dužinu i FFT limit).
"""

# -----------------------------------------------------------------------
# KONFIGURACIJA PUTANJE ZA IMPORT
# -----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from transmitter.ofdm import OFDMModulator
except ImportError as e:
    print("\n[GRESKA PRI IMPORTU]")
    print(f"Python ne može pronaći 'transmitter/ofdm.py'.")
    sys.exit(1)
# -----------------------------------------------------------------------

class TestExhaustiveOFDM(unittest.TestCase):
    """
    Testna klasa za verifikaciju LTE OFDM Modulatora.
    Sadrži 10 funkcionalnih testova.
    """

    def setUp(self):
        """Inicijalizacija parametara prije svakog testa."""
        self.ndlrb_list = [6, 15, 25, 50, 75, 100]
        self.base_fft_map = {
            6: 128, 15: 256, 25: 512, 50: 1024, 75: 1408, 100: 1408
        }

    def create_data(self, ndlrb, num_symbols):
        """Generiše random QAM simbole."""
        subcarriers = ndlrb * 12
        np.random.seed(42)
        return (np.random.randn(subcarriers, num_symbols) + 
                1j * np.random.randn(subcarriers, num_symbols))

    # =================================================================
    # GRUPA 1: OSNOVNE DIMENZIJE (Testovi 1-2)
    # =================================================================

    def test_01_sample_rate_correctness(self):
        """Test 1: Provjerava da li je Sample Rate ispravno izračunat."""
        print("\n[TEST 1] Provjera Sample Rate-a...")
        for ndlrb in self.ndlrb_list:
            grid = self.create_data(ndlrb, 14)
            mod = OFDMModulator(grid)
            expected_sr = self.base_fft_map[ndlrb] * 15000
            self.assertEqual(mod.sample_rate, expected_sr)
            
            # Test sa custom FFT-om
            mod_custom = OFDMModulator(grid, new_fft_size=2048)
            self.assertEqual(mod_custom.sample_rate, 2048 * 15000)

    def test_02_waveform_length_normal_cp(self):
        """Test 2: Provjerava dužinu izlaznog niza za Normal CP (7 simbola/slot)."""
        print("\n[TEST 2] Dužina niza (Normal CP)...")
        ndlrb = 25 # FFT 512
        grid = self.create_data(ndlrb, 14) # 1 subframe
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()
        
        # FFT=512. Normal CP struktura:
        # Simbol 0: CP=40 
        # Simboli 1-6: CP=36
        # Ukupno po slotu: 7*512 + 40 + 6*36 = 3840
        # Subframe (2 slota) = 7680
        self.assertEqual(len(wav), 7680)

    # =================================================================
    # GRUPA 2: INTEGRITET PODATAKA I DSP (Testovi 3-6)
    # =================================================================

    def test_03_cp_integrity_normal(self):
        """Test 3: Provjerava da li je CP kopija kraja simbola (Normal Mode)."""
        print("\n[TEST 3] CP Integritet (Normal)...")
        grid = self.create_data(25, 14)
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()
        
        # Provjera samo prvog simbola
        cp_len = mod.cp_lengths[0]
        N = mod.N
        
        cp_content = wav[0:cp_len]
        # Rep podataka je zadnjih cp_len uzoraka dijela sa podacima.
        data_part = wav[cp_len : cp_len + N]
        tail_part = data_part[-cp_len:]
        
        np.testing.assert_array_almost_equal(cp_content, tail_part)

    def test_04_cp_integrity_extended(self):
        """Test 4: Provjerava da li je CP kopija kraja simbola (Extended Mode)."""
        print("\n[TEST 4] CP Integritet (Extended)...")
        grid = self.create_data(25, 12)
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()
        
        cp_len = mod.cp_lengths[0]
        N = mod.N
        
        # Uzimamo drugi simbol (index 1)
        sym_len = N + cp_len
        start_idx = sym_len 
        
        cp_content = wav[start_idx : start_idx + cp_len]
        data_part = wav[start_idx + cp_len : start_idx + sym_len]
        tail_part = data_part[-cp_len:]
        
        np.testing.assert_array_almost_equal(cp_content, tail_part)

    def test_05_dsp_roundtrip_mse(self):
        """Test 5: Puni krug (Modulacija -> Demodulacija) provjera MSE."""
        print("\n[TEST 5] DSP Round-trip MSE...")
        ndlrb = 15
        grid_in = self.create_data(ndlrb, 14)
        mod = OFDMModulator(grid_in)
        wav, _ = mod.modulate()
        
        N = mod.N
        reconstructed = np.zeros_like(grid_in)
        current = 0
        
        pos_sc = np.arange(mod.num_subcarriers//2, mod.num_subcarriers)
        neg_sc = np.arange(0, mod.num_subcarriers//2)
        
        for i in range(14):
            cp = mod.cp_lengths[i % 7]
            data_t = wav[current + cp : current + cp + N]
            data_f = np.fft.fft(data_t) / N
            
            reconstructed[pos_sc, i] = data_f[0 : mod.num_subcarriers//2]
            reconstructed[neg_sc, i] = data_f[N - mod.num_subcarriers//2 : N]
            current += (cp + N)
            
        mse = np.mean(np.abs(grid_in - reconstructed)**2)
        self.assertLess(mse, 1e-20)

    def test_06_zero_input_response(self):
        """Test 6: Provjerava da li ulaz nula rezultira izlazom nula."""
        print("\n[TEST 6] Zero Input Response...")
        grid = np.zeros((300, 14), dtype=complex) # 25 RBs
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()
        self.assertEqual(np.sum(np.abs(wav)), 0.0)

    # =================================================================
    # GRUPA 3: SPECIJALNI SLUČAJEVI (Testovi 7-8)
    # =================================================================

    def test_07_single_active_subcarrier_papr(self):
        """
        Test 7: Single Tone Test.
        Jedan podnosioc -> konstantna magnituda u vremenu.
        """
        print("\n[TEST 7] Single Tone (Constant Envelope)...")
        grid = np.zeros((300, 14), dtype=complex)
        grid[0, 0] = 1.0 + 0j 
        
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()
        
        first_symbol_wav = wav[0 : mod.N + mod.cp_lengths[0]]
        magnitudes = np.abs(first_symbol_wav)
        std_dev = np.std(magnitudes)
        
        self.assertLess(std_dev, 1e-5)

    def test_08_oversampling_logic(self):
        """Test 8: Provjera da li 'new_fft_size' ispravno mijenja interno stanje."""
        print("\n[TEST 8] Oversampling logika...")
        grid = self.create_data(6, 14)
        target_fft = 512
        mod = OFDMModulator(grid, new_fft_size=target_fft)
        
        self.assertEqual(mod.N, target_fft)
        # Provjera skaliranja CP-a (Normal CP: 160 * 512/2048 = 40)
        expected_cp0 = int(160 * target_fft / 2048)
        self.assertEqual(mod.cp_lengths[0], expected_cp0)

    # =================================================================
    # GRUPA 4: ERROR HANDLING (Testovi 9-10)
    # =================================================================

    def test_09_error_invalid_subcarrier_count(self):
        """Test 9: Greška ako broj redova nije djeljiv sa 12."""
        print("\n[TEST 9] Error: Invalid Subcarrier Count...")
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((13, 14)))

    def test_10_error_invalid_symbol_count(self):
        """Test 10: Greška ako broj simbola nije 12 ili 14."""
        print("\n[TEST 10] Error: Invalid Symbol Count...")
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((72, 5)))

if __name__ == '__main__':
    unittest.main(verbosity=2)