import pytest
import numpy as np
from receiver.QPSK_demapiranje import QPSKDemapper

# ==============================================================================
# DEFINICIJA TESTNIH PODATAKA (SCENARIJI)
# Format: (ime_testa, ulazni_simboli, ocekivani_bitovi)
# ==============================================================================

test_scenarios = [
    # ---------------------------------------------------------
    # GRUPA 1: Idealne QPSK tačke (sredine kvadranata)
    # ---------------------------------------------------------
    # Logika mapiranja:
    # I > 0 -> b0=0, I <= 0 -> b0=1
    # Q > 0 -> b1=0, Q <= 0 -> b1=1
    # ---------------------------------------------------------
    ("1. Idealno [1+1j] (I>0, Q>0)",   [1+1j],   [0, 0]),
    ("2. Idealno [-1+1j] (I<0, Q>0)",  [-1+1j],  [1, 0]),
    ("3. Idealno [1-1j] (I>0, Q<0)",   [1-1j],   [0, 1]),
    ("4. Idealno [-1-1j] (I<0, Q<0)",  [-1-1j],  [1, 1]),

    # ---------------------------------------------------------
    # GRUPA 2: Tačke sa šumom (male varijacije)
    # ---------------------------------------------------------
    ("5. Sum [0.1+0.1j] (blizu centra)", [0.1+0.1j],   [0, 0]),
    ("6. Sum [-0.1-0.1j] (blizu centra)", [-0.1-0.1j], [1, 1]),
    ("7. Jaci signal [10+10j]",          [10+10j],     [0, 0]),
    ("8. Asimetricno [-5+0.5j]",         [-5+0.5j],    [1, 0]),

    # ---------------------------------------------------------
    # GRUPA 3: Granične vrijednosti (Boundary Conditions)
    # Korišten uslov (val <= 0) -> bit 1.
    # Dakle, tačna nula (0) se tumači kao bit 1.
    # ---------------------------------------------------------
    # I=0 (bit 1), Q=1 (bit 0) -> [1, 0]
    ("9. Granica I=0 [0+1j]",   [0+1j],   [1, 0]),
    
    # I=1 (bit 0), Q=0 (bit 1) -> [0, 1]
    ("10. Granica Q=0 [1+0j]",  [1+0j],   [0, 1]),
    
    # I=0 (bit 1), Q=0 (bit 1) -> [1, 1]
    ("11. Centar [0+0j]",       [0+0j],   [1, 1]),
    
    # I=-0.5 (bit 1), Q=0 (bit 1) -> [1, 1]
    ("12. Granica I<0, Q=0",    [-0.5+0j], [1, 1]),

    # ---------------------------------------------------------
    # GRUPA 4: Ekstremne vrijednosti
    # ---------------------------------------------------------
    ("13. Jako veliki brojevi", [1e6 - 1e6j], [0, 1]),
    ("14. Jako mali brojevi",   [1e-9 + 1e-9j], [0, 0]),

    # ---------------------------------------------------------
    # GRUPA 5: Vektorizacija (Više simbola odjednom)
    # ---------------------------------------------------------
    (
        "15. Niz od 2 simbola [1+1j, -1-1j]", 
        [1+1j, -1-1j], 
        [0, 0, 1, 1] # Prvi simbol 00, drugi 11
    ),
]

class TestQPSKDemapper:
    """
    Klasa za testiranje QPSKDemapper funkcionalnosti.
    Koristi pytest parametrizaciju za pokretanje 15 različitih slučajeva.
    """

    @pytest.mark.parametrize("ime_testa, ulaz, ocekivano", test_scenarios)
    def test_run_scenario(self, ime_testa, ulaz, ocekivano):
        """
        Ova funkcija se izvršava automatski za svaki red u test_scenarios.
        
        Parametri
        ----------
        ime_testa : str
            Opis testa (ispisuje se u terminalu)
        ulaz : list
            Lista kompleksnih brojeva (simboli)
        ocekivano : list
            Lista očekivanih bitova (0 ili 1)
        """
        
        # 1. Inicijalizacija demappera
        demapper = QPSKDemapper(mode="hard")

        # 2. Priprema NumPy nizova (osiguravamo tipove podataka)
        simboli = np.array(ulaz, dtype=np.complex64)
        ocekivani_bitovi = np.array(ocekivano, dtype=np.uint8)

        # 3. Izvršavanje demapiranja
        dobiveni_bitovi = demapper.demap(simboli)

        # 4. Provjera ispravnosti (Assert)
        # Koristimo numpy.testing funkciju koja daje detaljan ispis ako nizovi nisu isti
        np.testing.assert_array_equal(
            dobiveni_bitovi, 
            ocekivani_bitovi, 
            err_msg=f"GREŠKA u testu: '{ime_testa}'"
        )