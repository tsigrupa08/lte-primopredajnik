import numpy as np
from receiver.QPSK_demapiranje import QPSKDemapper  

def test_QPSKDemapper():
    """
    Unit testovi za QPSKDemapper
    Pokrivaju:
    - idealne QPSK tačke
    - simboli sa malim šumom
    - granične vrijednosti (I=0 ili Q=0)
    - veći broj simbola odjednom
    """
    demapper = QPSKDemapper(mode="hard")

    # Lista test slučajeva: (simboli, očekivani bitovi)
    test_cases = [
        # Idealne tačke
        (np.array([1 + 1j], dtype=np.complex64), [0, 0]),
        (np.array([-1 + 1j], dtype=np.complex64), [1, 0]),
        (np.array([1 - 1j], dtype=np.complex64), [0, 1]),
        (np.array([-1 - 1j], dtype=np.complex64), [1, 1]),

        # Male varijacije oko idealnih
        (np.array([0.8 + 1.2j], dtype=np.complex64), [0, 0]),
        (np.array([-1.1 + 0.9j], dtype=np.complex64), [1, 0]),
        (np.array([0.7 - 1.3j], dtype=np.complex64), [0, 1]),
        (np.array([-0.6 - 0.8j], dtype=np.complex64), [1, 1]),

        # Granične vrijednosti
        (np.array([0 + 1j], dtype=np.complex64), [1, 0]),    # I=0 → b0=1
        (np.array([1 + 0j], dtype=np.complex64), [0, 1]),    # Q=0 → b1=1
        (np.array([0 - 1j], dtype=np.complex64), [1, 1]),    # I=0, Q<0
        (np.array([-0.5 + 0j], dtype=np.complex64), [1, 1]), # I<0, Q=0
        (np.array([0 + 0j], dtype=np.complex64), [1, 1]),    # I=0, Q=0

        # Više simbola odjednom
        (np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=np.complex64),
         [0,0, 1,0, 0,1, 1,1])
    ]

    # Pokretanje testova
    for i, (symbols, expected) in enumerate(test_cases, 1):
        bits = demapper.demap(symbols)
        if np.array_equal(bits, np.array(expected, dtype=np.uint8)):
            print(f"Test {i}: PASS")
        else:
            print(f"Test {i}: FAIL - Dobio {bits.tolist()}, očekivano {expected}")

if __name__ == "__main__":
    test_QPSKDemapper()
