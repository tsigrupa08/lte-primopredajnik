import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from transmitter.pss import PSSGenerator
from transmitter.resource_grid import ResourceGrid
from transmitter.ofdm import OFDMModulator
from channel.awgn_channel import AWGNChannel

# ================================================================
# Results folders
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(BASE_DIR, "results")

TX_DIR = os.path.join(RESULTS_BASE, "tx")
CH_DIR = os.path.join(RESULTS_BASE, "channel")
RX_DIR = os.path.join(RESULTS_BASE, "rx")

os.makedirs(TX_DIR, exist_ok=True)
os.makedirs(CH_DIR, exist_ok=True)
os.makedirs(RX_DIR, exist_ok=True)

# ================================================================
# Parametri
# ================================================================
ndlrb = 6
num_subframes = 1
pss_symbol_idx = 6
pbch_symbol_idx = 0
snr_db = 10
fft_size = 128

# ================================================================
# 1. PSS signal (TX)
# ================================================================
pss_signal = PSSGenerator.generate(1)

# ================================================================
# 2. PBCH QPSK simboli (TX)
# ================================================================
pbch_symbols = np.array([
    1 + 1j,
    -1 + 1j,
    -1 - 1j,
    1 - 1j
]) / np.sqrt(2)
pbch_symbols = np.tile(pbch_symbols, 50)

# ================================================================
# 3. Resource grid (TX)
# ================================================================
grid = ResourceGrid(
    ndlrb=ndlrb,
    num_subframes=num_subframes,
    normal_cp=True
)
grid.map_pss(pss_sequence=pss_signal, symbol_index=pss_symbol_idx)
grid.map_pbch(pbch_symbols=pbch_symbols,
              pbch_symbol_indices=[pbch_symbol_idx])

# ================================================================
# 4. OFDM modulacija (TX)
# ================================================================
ofdm = OFDMModulator(resource_grid=grid.grid, new_fft_size=fft_size)
tx_signal, fs = ofdm.modulate()
tx_signal = tx_signal.astype(np.complex64)
tx_signal -= np.mean(tx_signal)

# ================================================================
# 5. AWGN kanal (CHANNEL)
# ================================================================
awgn = AWGNChannel(snr_db=snr_db, seed=42)
rx_signal = awgn.apply(tx_signal)
rx_signal -= np.mean(rx_signal)

plot_len = min(2000, len(tx_signal))

# ================================================================
# TX: PBCH konstelacija
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(pbch_symbols.real, pbch_symbols.imag, s=40)
plt.title("PBCH QPSK konstelacija (TX)")
plt.grid(True)
plt.axis("equal")
plt.savefig(os.path.join(TX_DIR, "pbch_constellation_tx.png"), dpi=300)
plt.close()

# ================================================================
# CHANNEL: PBCH konstelacija + AWGN
# ================================================================
noise = rx_signal[:len(pbch_symbols)]
plt.figure(figsize=(6, 6))
plt.scatter(noise.real, noise.imag, s=40)
plt.title("PBCH konstelacija (AWGN kanal)")
plt.grid(True)
plt.axis("equal")
plt.savefig(os.path.join(CH_DIR, "pbch_constellation_awgn.png"), dpi=300)
plt.close()

# ================================================================
# TX: OFDM konstelacija
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(tx_signal[:plot_len].real,
            tx_signal[:plot_len].imag, s=5)
plt.title("OFDM konstelacija (TX)")
plt.grid(True)
plt.axis("equal")
plt.savefig(os.path.join(TX_DIR, "ofdm_constellation_tx.png"), dpi=300)
plt.close()

# ================================================================
# RX: OFDM konstelacija
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(rx_signal[:plot_len].real,
            rx_signal[:plot_len].imag, s=5)
plt.title("OFDM konstelacija (RX, AWGN)")
plt.grid(True)
plt.axis("equal")
plt.savefig(os.path.join(RX_DIR, "ofdm_constellation_rx.png"), dpi=300)
plt.close()

# ================================================================
# RX: OFDM faza
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.angle(rx_signal[:plot_len]))
plt.title("OFDM faza (RX, AWGN)")
plt.grid(True)
plt.savefig(os.path.join(RX_DIR, "ofdm_phase_rx.png"), dpi=300)
plt.close()

# ================================================================
# TX / RX: realni dio signala
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(tx_signal[:plot_len].real, label="TX")
plt.plot(rx_signal[:plot_len].real, label="RX", alpha=0.7)
plt.title("OFDM realni dio (TX vs RX)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RX_DIR, "ofdm_real_tx_rx.png"), dpi=300)
plt.close()

print("[OK] AWGN example završen.")
print("Slike su snimljene u examples/results/{tx,channel,rx}/")
