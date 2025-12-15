import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from transmitter.pss import PSSGenerator
from transmitter.resource_grid import ResourceGrid
from transmitter.pbch import PBCHEncoder
from transmitter.ofdm import OFDMModulator
from channel.awgn_channel import AWGNChannel

# ------------------------------------------------
# Parametri
# ------------------------------------------------
ndlrb = 6
num_subframes = 1
pss_symbol_idx = 6
pbch_symbol_idx = 0
snr_db = 10

# ------------------------------------------------
# 1. PSS signal
# ------------------------------------------------
pss_signal = PSSGenerator.generate(1)

# ------------------------------------------------
# 2. PBCH simboli (QPSK)
# ------------------------------------------------
pbch_symbols_ideal = np.array([
    1 + 1j,
    -1 + 1j,
    -1 - 1j,
    1 - 1j
]) / np.sqrt(2)

pbch_symbols = np.tile(pbch_symbols_ideal, 50)

# AWGN na PBCH
sigma_pbch = np.sqrt(np.mean(np.abs(pbch_symbols)**2) / (10**(snr_db/10) * 2))
rng = np.random.default_rng(42)
pbch_symbols_awgn = pbch_symbols + sigma_pbch * (rng.standard_normal(len(pbch_symbols)) + 1j*rng.standard_normal(len(pbch_symbols)))

# ------------------------------------------------
# 3. Resource grid
# ------------------------------------------------
grid = ResourceGrid(ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=True)
grid.map_pss(pss_sequence=pss_signal, symbol_index=pss_symbol_idx)
grid.map_pbch(pbch_symbols=pbch_symbols, pbch_symbol_indices=[pbch_symbol_idx])

# ------------------------------------------------
# 4. OFDM modulacija
# ------------------------------------------------
fft_size = 128
ofdm = OFDMModulator(resource_grid=grid.grid, new_fft_size=fft_size)
tx_signal, sample_rate = ofdm.modulate()
tx_signal = tx_signal.astype(np.complex64)

# centriranje OFDM signala
tx_signal -= np.mean(tx_signal)

# ------------------------------------------------
# 5. AWGN kanal
# ------------------------------------------------
awgn = AWGNChannel(snr_db=snr_db, seed=42)
rx_signal = awgn.apply(tx_signal)

# centriranje RX OFDM signala
rx_signal -= np.mean(rx_signal)

plot_len = min(2000, len(tx_signal))

# ------------------------------------------------
# 6. Konstelacije
# ------------------------------------------------
# PBCH QPSK: spojen TX i RX
plt.figure(figsize=(6,6))
plt.scatter(np.real(pbch_symbols), np.imag(pbch_symbols), s=40, c='blue', label="PBCH TX")
plt.scatter(np.real(pbch_symbols_awgn), np.imag(pbch_symbols_awgn), s=40, c='red', label="PBCH RX (AWGN)")
plt.title("PBCH QPSK Konstelacija")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig("constellation_pbch_combined.png", dpi=300)
plt.close()

# OFDM konstelacija TX
plt.figure(figsize=(6,6))
plt.scatter(np.real(tx_signal[:plot_len]), np.imag(tx_signal[:plot_len]), s=5, c='green', label="OFDM TX")
plt.title("OFDM Konstelacija TX (prije AWGN)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.legend()
plt.savefig("constellation_ofdm_tx.png", dpi=300)
plt.close()

# OFDM konstelacija RX
plt.figure(figsize=(6,6))
plt.scatter(np.real(rx_signal[:plot_len]), np.imag(rx_signal[:plot_len]), s=5, c='orange', label="OFDM RX (poslije AWGN)")
plt.title("OFDM Konstelacija RX (poslije AWGN)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.legend()
plt.savefig("constellation_ofdm_rx.png", dpi=300)
plt.close()

# ------------------------------------------------
# 7. PBCH faza
# ------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(np.angle(pbch_symbols), label="PBCH Faza TX (prije AWGN)")
plt.title("PBCH Faza prije AWGN")
plt.xlabel("Simbol")
plt.ylabel("Faza [rad]")
plt.grid(True)
plt.legend()
plt.savefig("phase_pbch_before_awgn.png", dpi=300)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(np.angle(pbch_symbols_awgn), label=f"PBCH Faza RX (poslije AWGN, {snr_db} dB)", color='red')
plt.title("PBCH Faza poslije AWGN")
plt.xlabel("Simbol")
plt.ylabel("Faza [rad]")
plt.grid(True)
plt.legend()
plt.savefig("phase_pbch_after_awgn.png", dpi=300)
plt.close()

# ------------------------------------------------
# 8. OFDM faza
# ------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(np.angle(rx_signal[:plot_len]), label=f"OFDM Faza RX (poslije AWGN, {snr_db} dB)", color='orange')
plt.title("OFDM Faza poslije AWGN")
plt.xlabel("Uzorak")
plt.ylabel("Faza [rad]")
plt.grid(True)
plt.legend()
plt.savefig("phase_ofdm_rx.png", dpi=300)
plt.close()

# ------------------------------------------------
# 9. Realni dio OFDM signala
# ------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(np.real(tx_signal[:plot_len]), label="OFDM Realni TX (prije AWGN)")
plt.title("OFDM Realni dio TX")
plt.xlabel("Uzorak")
plt.ylabel("Amplituda")
plt.grid(True)
plt.legend()
plt.savefig("real_ofdm_tx.png", dpi=300)
plt.close()

plt.figure(figsize=(10,4))
plt.plot(np.real(rx_signal[:plot_len]), label="OFDM Realni RX (poslije AWGN)", color='purple')
plt.title("OFDM Realni dio RX")
plt.xlabel("Uzorak")
plt.ylabel("Amplituda")
plt.grid(True)
plt.legend()
plt.savefig("real_ofdm_rx.png", dpi=300)
plt.close()

plt.show()
