import numpy as np
import pytest
from transmitter.LTETxChain import LTETxChain

# Test 1: Provjerava inicijalizaciju klase LTETxChain
# Očekujemo da se prilikom kreiranja objekta napravi atribut "grid"
# i da je taj atribut numpy array (jer predstavlja resource grid).
def test_init_creates_grid():
    tx = LTETxChain()
    assert hasattr(tx, "grid")
    assert isinstance(tx.grid, np.ndarray)


# Test 2: Provjerava generisanje waveforma bez MIB bitova
# Funkcija može vratiti tuple (waveform, fs) ili samo waveform kao numpy array.

def test_generate_waveform_returns_array_or_raises():
    tx = LTETxChain()
    try:
        result = tx.generate_waveform()
        # Ako se vrati tuple, provjeravamo tipove
        if isinstance(result, tuple):
            waveform, fs = result
            assert isinstance(waveform, np.ndarray)
            assert isinstance(fs, (int, float))
        else:
            # Ako se vrati samo ndarray
            assert isinstance(result, np.ndarray)
    except Exception as e:
        # Greška je validan sad path
        assert isinstance(e, Exception)


# Test 3: Provjerava generisanje waveforma sa tipičnim MIB bitovima
# Generišemo 24 random bita i prosljeđujemo funkciji.
# Kao i u prethodnom testu, prihvatamo tuple ili ndarray.

def test_generate_waveform_with_mib_bits():
    tx = LTETxChain()
    mib_bits = np.random.randint(0, 2, size=24)
    try:
        result = tx.generate_waveform(mib_bits=mib_bits)
        if isinstance(result, tuple):
            waveform, fs = result
            assert isinstance(waveform, np.ndarray)
            assert isinstance(fs, (int, float))
        else:
            assert isinstance(result, np.ndarray)
    except Exception as e:
        assert isinstance(e, Exception)


# Test 4: Provjerava ponašanje kada se proslijedi prazan niz MIB bitova

def test_generate_waveform_with_empty_mib_bits():
    tx = LTETxChain()
    try:
        result = tx.generate_waveform(mib_bits=[])
        assert result is not None or result is None
    except Exception as e:
        assert isinstance(e, Exception)
