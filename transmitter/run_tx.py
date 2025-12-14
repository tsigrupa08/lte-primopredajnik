"""
Primjer pozivanja LTE transmit chain i OFDM modulatora.
"""

import numpy as np
from transmitter.LTETxChain import LTETxChain
from transmitter.ofdm import OFDMModulator

# Reproducibilnost
np.random.seed(0)

# -----------------------------------------------
# 1. Korištenje LTETxChain
# -----------------------------------------------
tx = LTETxChain(n_id_2=0, ndlrb=6, num_subframes=1, normal_cp=True)
mib_bits = np.random.randint(0, 2, 1920)
waveform, fs = tx.generate_waveform(mib_bits)

print("[LTETxChain] OFDM waveform generisan")
print("Duzina signala:", len(waveform))
print("Shape signala:", waveform.shape)
print("Sample rate:", fs)

# -----------------------------------------------
# 2. Direktno korištenje OFDMModulatora
# -----------------------------------------------
resource_grid = np.random.randn(72, 14) + 1j*np.random.randn(72, 14)
ofdm = OFDMModulator(resource_grid)
ofdm_waveform, ofdm_fs = ofdm.modulate()

print("[OFDMModulator] OFDM waveform generisan direktno")
print("Duzina signala:", len(ofdm_waveform))
print("Shape signala:", ofdm_waveform.shape)
print("Sample rate:", ofdm_fs)
