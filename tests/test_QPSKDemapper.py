import pytest
import numpy as np
from receiver.QPSK_demapiranje import QPSKDemapper

# ==============================================================================
# DEFINICIJA PODATAKA ZA TESTIRANJE
# ==============================================================================
# Ovde definišemo sve slučajeve.
# Svaki red je jedan test: (ime_testa, ulazni_simboli, ocekivani_bitovi)

test_data = [
    # --- IDEALNE TAČKE ---
    ("1. Idealno [1+1j]",   [1+1j],   [0, 0]),
    ("2. Idealno [-1+1j]",  [-1+1j],  [1, 0]),
    ("3. Idealno [1-1j]",   [1-1j],   [0, 1]),
    ("4. Idealno [-1-1j]",  [-1-1j],  [1, 1]),

    # --- MALE VARIJACIJE (ŠUM) ---
    ("5. Sum [0.8+1.2j]",   [0.8+1.2j],   [0, 0]),
    ("6. Sum [-1.1+0.9j]",  [-1.1+0.9j],  [1, 0]),
    ("7. Sum [0.7-1.3j]",   [0.7-1.3j],   [0, 1]),
    ("8. Sum [-0.6-0.8j]",  [-0.6-0.8j],  [1, 1]),

    # --- GRANIČNE VREDNOSTI (ZERO BOUNDARY) ---
    # Tvoja logika kaze: (val <= 0) -> bit 1. Dakle 0 je bit 1.
    ("9. Granica I=0 [0+1j]",   [0+1j],   [1, 0]),
    ("10. Granica Q=0 [1+0j]",  [1+0j],   [0, 1]),
    ("11. Centar [0+0j]",       [0+0j],   [1, 1]),
    ("12. Granica I<0 [-0.5+0j]", [-0.5+0j], [1, 1]),

    # --- VEKTORIZACIJA (VIŠE SIMBOLA) ---
    (
        "13. Niz od 4 simbola", 
        [1+1j, -1+1j, 1-1j, -1-1j], 
        [0,0,  1,0,  0,1,  1,1]
    ),
    
    # --- EKSTREMNE VREDNOSTI ---
    ("14. Veliki brojevi [100-100j]", [100-100j], [0, 1]),
    ("15. Mali brojevi [0.001+0.001j]", [0.001+0.001j], [0, 0]),
]

# ==============================================================================
# TEST FUNKCIJA
# ==============================================================================
@pytest.mark.parametrize("ime_testa, ulaz, ocekivano", test_data)
def test_QPSK_cases(ime_testa, ulaz, ocekivano):
    """
    Ova funkcija se automatski pokreće za svaki red u test_data.
    """
    # 1. Inicijalizacija
    demapper = QPSKDemapper(mode="hard")
    
    # Konverzija ulaza u numpy niz (za svaki slučaj)
    simboli = np.array(ulaz, dtype=np.complex64)
    ocekivani_bitovi = np.array(ocekivano, dtype=np.uint8)

    # 2. Izvršenje
    izlazni_bitovi = demapper.demap(simboli)

    # 3. Provera (Assert)
    # Ako ovo padne, pytest će ispisati tačno koji test je pao
    np.testing.assert_array_equal(
        izlazni_bitovi, 
        ocekivani_bitovi, 
        err_msg=f"GRESKA u testu '{ime_testa}'"
    )