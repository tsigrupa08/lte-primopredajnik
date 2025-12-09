import unittest
import numpy as np
import sys
import os

"""
Modul za testiranje OFDM Modulatora.

Ovaj modul implementira 'exhaustive' (iscrpno) testiranje klase OFDMModulator.
Testovi pokrivaju sve standardne LTE širine kanala, tipove cikličkog prefiksa
i validaciju DSP lanca.

Primjer pokretanja
------------------
Iz root foldera projekta:
    $ python tests/test_ofdm.py
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

    Ova klasa nasljeđuje `unittest.TestCase` i izvodi seriju testova
    kako bi osigurala matematičku tačnost modulacije.

    Attributes
    ----------
    ndlrb_list : list of int
        Lista podržanih LTE širina kanala (broj Resource Blokova).
    base_fft_map : dict
        Mapiranje između broja RB-ova i minimalne FFT veličine.
    all_fft_sizes : list of int
        Sve dozvoljene FFT veličine za testiranje oversamplinga.
    """

    def setUp(self):
        """
        Inicijalizacija testnih parametara prije svakog testa.
        Definiše standardne LTE konfiguracije.
        """
        # SVI validni LTE bandwidthi
        self.ndlrb_list = [6, 15, 25, 50, 75, 100]
        
        # Mapiranje NDLRB -> Minimalni FFT
        self.base_fft_map = {
            6: 128, 15: 256, 25: 512, 50: 1024, 75: 1408, 100: 1408
        }
        
        # SVE validne FFT veličine
        self.all_fft_sizes = [128, 256, 512, 1024, 1408, 1536, 2048]

    def create_data(self, ndlrb, num_symbols):
        """
        Pomoćna metoda za generisanje testnih podataka.

        Generiše matricu nasumičnih kompleksnih brojeva (QAM simbola)
        koja simulira ulazni Resource Grid.

        Parameters
        ----------
        ndlrb : int
            Broj downlink resource blokova (određuje širinu frekvencije).
        num_symbols : int
            Broj OFDM simbola u vremenu (kolone matrice).

        Returns
        -------
        np.ndarray
            Kompleksna matrica dimenzija (ndlrb * 12, num_symbols).
        """
        subcarriers = ndlrb * 12
        np.random.seed(42) # Deterministički seed za ponovljivost
        return (np.random.randn(subcarriers, num_symbols) + 
                1j * np.random.randn(subcarriers, num_symbols))

    def test_01_matrix_dimensions_check(self):
        """
        Verifikuje dimenzije izlaznog talasnog oblika (Waveform).

        Iterira kroz sve kombinacije NDLRB, tipova CP-a i FFT veličina.
        Provjerava da li izlazni niz ima tačan broj uzoraka prema formuli:
        Length = Sum(N_fft + N_cp_i) za sve simbole.

        Raises
        ------
        AssertionError
            Ako izračunata dužina niza ne odgovara teoretskoj dužini.
        """
        print(f"\n[TEST 1] Provjera dimenzija za sve kombinacije...")
        count = 0
        
        for ndlrb in self.ndlrb_list:
            min_fft = self.base_fft_map[ndlrb]
            # Testiramo default FFT i sve veće validne FFT-ove
            valid_ffts = [None] + [f for f in self.all_fft_sizes if f >= min_fft]

            for fft_size in valid_ffts:
                for symbols_per_subframe, mode_name in [(14, "Normal"), (12, "Extended")]:
                    
                    with self.subTest(ndlrb=ndlrb, fft=fft_size, mode=mode_name):
                        grid = self.create_data(ndlrb, symbols_per_subframe)
                        mod = OFDMModulator(grid, new_fft_size=fft_size)
                        
                        waveform, sr = mod.modulate()
                        
                        # Provjera sample rate-a
                        expected_N = fft_size if fft_size else min_fft
                        self.assertEqual(sr, expected_N * 15000)
                        
                        # Provjera dužine niza
                        multiplier = expected_N // 128
                        if mode_name == "Normal":
                            # Slot struktura: 1. simbol ima duži CP, ostalih 6 kraći
                            cp_slot = (10 * multiplier) + 6 * (9 * multiplier)
                            samples_per_slot = 7 * expected_N + cp_slot
                        else:
                            # Extended struktura: 6 simbola, svi isti CP
                            cp_slot = 6 * (32 * multiplier)
                            samples_per_slot = 6 * expected_N + cp_slot
                            
                        expected_len = 2 * samples_per_slot
                        self.assertEqual(len(waveform), expected_len)
                        count += 1
        print(f" -> Verifikovano {count} konfiguracija.")

    def test_02_exhaustive_cp_integrity(self):
        """
        Verifikuje sadržaj Cikličkog Prefiksa (Bit-to-Bit).

        Za svaki generisani OFDM simbol provjerava da li je CP na početku
        simbola identična kopija kraja tog simbola (rep podataka).

        Logic
        -----
        Za svaki simbol `i`:
            CP = waveform[start : start + cp_len]
            Tail = waveform[start + cp_len + N - cp_len : start + cp_len + N]
            Assert CP == Tail
        """
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
                        
                        # Izdvajanje CP-a i repa podataka
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
        """
        Izvodi DSP Round-trip test (Modulacija -> Ručna Demodulacija).

        Simulira idealni kanal (bez šuma) i provjerava da li se ulazni podaci
        mogu rekonstruisati nakon IFFT, dodavanja CP-a i uklanjanja istog.

        Verifikuje:
        1. Mapiranje podnosioca (pozitivne/negativne frekvencije).
        2. IFFT/FFT skaliranje.
        3. CP logiku.

        Raises
        ------
        AssertionError
            Ako je srednja kvadratna greška (MSE) između ulaza i izlaza > 1e-20.
        """
        print(f"\n[TEST 3] DSP Round-trip verifikacija...")
        
        for ndlrb in self.ndlrb_list:
            with self.subTest(ndlrb=ndlrb):
                grid_in = self.create_data(ndlrb, 14)
                mod = OFDMModulator(grid_in)
                waveform, _ = mod.modulate()
                
                N = mod.N
                grid_out = np.zeros_like(grid_in)
                
                # Rekonstrukcija mapiranja indeksa
                pos_sc = np.arange(mod.num_subcarriers//2, mod.num_subcarriers)
                neg_sc = np.arange(0, mod.num_subcarriers//2)
                pos_freq = np.arange(0, mod.num_subcarriers//2)
                neg_freq = np.arange(N - mod.num_subcarriers//2, N)
                
                current_idx = 0
                for i in range(14):
                    sym_in_slot = i % 7
                    cp_len = mod.cp_lengths[sym_in_slot]
                    
                    # Skidanje CP-a i FFT
                    fft_input = waveform[current_idx + cp_len : current_idx + cp_len + N]
                    freq_domain = np.fft.fft(fft_input) / N
                    
                    # Mapiranje nazad u grid
                    grid_out[pos_sc, i] = freq_domain[pos_freq]
                    grid_out[neg_sc, i] = freq_domain[neg_freq]
                    
                    current_idx += (cp_len + N)
                
                mse = np.mean(np.abs(grid_in - grid_out)**2)
                self.assertLess(mse, 1e-20)
        print(" -> DSP logika ispravna.")

    def test_04_multi_subframe(self):
        """
        Testira generisanje dužih sekvenci (Multi-subframe).

        Provjerava stabilnost i ispravnost dužine signala za 10ms okvir
        (1 Radio Frame = 10 Subframes).

        Parameters
        ----------
        None (Hardkodiran NDLRB=25, 140 simbola).
        """
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
        """
        Testira rukovanje greškama (Error Handling).

        Provjerava da li klasa ispravno podiže `ValueError` za nevalidne ulaze.

        Cases
        -----
        1. Broj podnosioca nije djeljiv sa 12.
        2. Broj simbola ne odgovara LTE slot strukturi.
        3. Nevalidan override FFT veličine.
        4. Nepodržan NDLRB.
        """
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