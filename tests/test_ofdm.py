import unittest
import numpy as np
import sys
import os

# -----------------------------------------------------------------------
# KONFIGURACIJA PUTANJE ZA IMPORT (CRITICAL STEP)
# -----------------------------------------------------------------------
# Ovo omogućava da test vidi 'transmitter' paket iako je u drugom folderu.
# 1. Uzimamo putanju gdje se nalazi ovaj fajl (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Idemo jedan korak nazad do root foldera (LTE-PRIMOPREDAJNIK/)
project_root = os.path.dirname(current_dir)
# 3. Dodajemo root u sistemsku putanju
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Sada možemo importovati tvoju klasu tačno onako kako je u strukturi
try:
    from transmitter.ofdm import OFDMModulator
except ImportError as e:
    print("\n[GRESKA PRI IMPORTU]")
    print(f"Python ne može pronaći 'transmitter/ofdm.py'.")
    print(f"Trenutna putanja: {sys.path[0]}")
    print(f"Detalji greške: {e}")
    sys.exit(1)
# -----------------------------------------------------------------------

class TestExhaustiveOFDM(unittest.TestCase):

    def setUp(self):
        # SVI validni LTE bandwidthi
        self.ndlrb_list = [6, 15, 25, 50, 75, 100]
        
        # Mapiranje NDLRB -> Minimalni FFT
        self.base_fft_map = {
            6: 128, 15: 256, 25: 512, 50: 1024, 75: 1408, 100: 1408
        }
        
        # SVE validne FFT veličine
        self.all_fft_sizes = [128, 256, 512, 1024, 1408, 1536, 2048]

    def create_data(self, ndlrb, num_symbols):
        """Generiše random QAM simbole."""
        subcarriers = ndlrb * 12
        np.random.seed(42) # Deterministički seed za testove
        return (np.random.randn(subcarriers, num_symbols) + 
                1j * np.random.randn(subcarriers, num_symbols))

    def test_01_matrix_dimensions_check(self):
        """[TEST 1] Matrični test dimenzija (Sve kombinacije RB, CP, FFT)"""
        print(f"\n[TEST 1] Provjera dimenzija za sve kombinacije...")
        count = 0
        
        for ndlrb in self.ndlrb_list:
            min_fft = self.base_fft_map[ndlrb]
            valid_ffts = [None] + [f for f in self.all_fft_sizes if f >= min_fft]

            for fft_size in valid_ffts:
                for symbols_per_subframe, mode_name in [(14, "Normal"), (12, "Extended")]:
                    
                    with self.subTest(ndlrb=ndlrb, fft=fft_size, mode=mode_name):
                        grid = self.create_data(ndlrb, symbols_per_subframe)
                        mod = OFDMModulator(grid, new_fft_size=fft_size)
                        
                        waveform, sr = mod.modulate()
                        
                        expected_N = fft_size if fft_size else min_fft
                        self.assertEqual(sr, expected_N * 15000)
                        
                        multiplier = expected_N // 128
                        if mode_name == "Normal":
                            cp_slot = (10 * multiplier) + 6 * (9 * multiplier)
                            samples_per_slot = 7 * expected_N + cp_slot
                        else:
                            cp_slot = 6 * (32 * multiplier)
                            samples_per_slot = 6 * expected_N + cp_slot
                            
                        expected_len = 2 * samples_per_slot
                        self.assertEqual(len(waveform), expected_len)
                        count += 1
        print(f" -> Verifikovano {count} konfiguracija.")

    def test_02_exhaustive_cp_integrity(self):
        """[TEST 2] Bit-to-bit provjera Cikličkog Prefiksa"""
        print(f"\n[TEST 2] Provjera integriteta CP-a...")
        
        for ndlrb in self.ndlrb_list:
            for is_normal_cp in [True, False]:
                n_symbols = 14 if is_normal_cp else 12
                mode = "Normal" if is_normal_cp else "Extended"
                
                with self.subTest(ndlrb=ndlrb, mode=mode):
                    grid = self.create_data(ndlrb, n_symbols)
                    mod = OFDMModulator(grid)
                    waveform, _ = mod.modulate()
                    
                    current_idx = 0
                    N = mod.N
                    
                    for sym_idx in range(n_symbols):
                        sym_in_slot = sym_idx % mod.n_symbols_per_slot
                        cp_len = mod.cp_lengths[sym_in_slot]
                        
                        extracted_cp = waveform[current_idx : current_idx + cp_len]
                        
                        data_start = current_idx + cp_len
                        data_end = data_start + N
                        extracted_data = waveform[data_start : data_end]
                        data_tail = extracted_data[-cp_len:]
                        
                        np.testing.assert_array_almost_equal(extracted_cp, data_tail,
                            err_msg=f"CP mismatch: NDLRB={ndlrb}, Sym={sym_idx}")
                        
                        current_idx += (cp_len + N)
        print(" -> CP integritet potvrđen.")

    def test_03_dsp_roundtrip_verification(self):
        """[TEST 3] Modulacija -> Demodulacija (Round-trip)"""
        print(f"\n[TEST 3] DSP Round-trip verifikacija...")
        
        for ndlrb in self.ndlrb_list:
            with self.subTest(ndlrb=ndlrb):
                grid_in = self.create_data(ndlrb, 14)
                mod = OFDMModulator(grid_in)
                waveform, _ = mod.modulate()
                
                N = mod.N
                grid_out = np.zeros_like(grid_in)
                
                pos_sc = np.arange(mod.num_subcarriers//2, mod.num_subcarriers)
                neg_sc = np.arange(0, mod.num_subcarriers//2)
                pos_freq = np.arange(0, mod.num_subcarriers//2)
                neg_freq = np.arange(N - mod.num_subcarriers//2, N)
                
                current_idx = 0
                for i in range(14):
                    sym_in_slot = i % 7
                    cp_len = mod.cp_lengths[sym_in_slot]
                    
                    fft_input = waveform[current_idx + cp_len : current_idx + cp_len + N]
                    freq_domain = np.fft.fft(fft_input) / N
                    
                    grid_out[pos_sc, i] = freq_domain[pos_freq]
                    grid_out[neg_sc, i] = freq_domain[neg_freq]
                    
                    current_idx += (cp_len + N)
                
                mse = np.mean(np.abs(grid_in - grid_out)**2)
                self.assertLess(mse, 1e-20)
        print(" -> DSP logika ispravna.")

    def test_04_multi_subframe(self):
        """[TEST 4] Generisanje 10ms okvira (10 subframe-ova)"""
        print(f"\n[TEST 4] Multi-subframe test...")
        ndlrb = 25
        num_symbols = 140 # 10 subframes
        grid = self.create_data(ndlrb, num_symbols)
        
        mod = OFDMModulator(grid)
        wav, _ = mod.modulate()
        
        expected = 76800 # Za 5MHz (FFT 512)
        self.assertEqual(len(wav), expected)
        print(" -> Multi-subframe OK.")

    def test_05_invalid_configurations(self):
        """[TEST 5] Validacija grešaka"""
        print(f"\n[TEST 5] Error Handling...")
        
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((13, 14))) 
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((72, 13)))
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((600, 14)), new_fft_size=123)
        with self.assertRaises(ValueError):
            OFDMModulator(np.zeros((20*12, 14)))
            
        print(" -> Error handling OK.")

if __name__ == '__main__':
    unittest.main(verbosity=2)