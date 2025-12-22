import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from transmitter.pss import PSSGenerator
from transmitter.resource_grid import ResourceGrid
from transmitter.pbch import PBCHEncoder
from transmitter.ofdm import OFDMModulator
from channel.awgn_channel import AWGNChannel

# ================================================================
# Results folder: examples/results
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
# Parametri
# ================================================================
ndlrb = 6
num_subframes = 1
pss_symbol_idx = 6
pbch_symbol_idx = 0
snr_db = 10

# ================================================================
# 1. PSS signal
# ================================================================
pss_signal = PSSGenerator.generate(1)

# ================================================================
# 2. PBCH simboli (QPSK)
# ================================================================
pbch_symbols_ideal = np.array([
    1 + 1j,
    -1 + 1j,
    -1 - 1j,
    1 - 1j
]) / np.sqrt(2)

pbch_symbols = np.tile(pbch_symbols_ideal, 50)

# AWGN na PBCH (za konstelaciju)
sigma_pbch = np.sqrt(
    np.mean(np.abs(pbch_symbols) ** 2) / (10 ** (snr_db / 10) * 2)
)
rng = np.random.default_rng(42)
pbch_symbols_awgn = pbch_symbols + sigma_pbch * (
    rng.standard_normal(len(pbch_symbols)) +
    1j * rng.standard_normal(len(pbch_symbols))
)

# ================================================================
# 3. Resource grid
# ================================================================
grid = ResourceGrid(
    ndlrb=ndlrb,
    num_subframes=num_subframes,
    normal_cp=True
)
grid.map_pss(pss_sequence=pss_signal, symbol_index=pss_symbol_idx)
grid.map_pbch(
    pbch_symbols=pbch_symbols,
    pbch_symbol_indices=[pbch_symbol_idx]
)

# ================================================================
# 4. OFDM modulacija
# ================================================================
fft_size = 128
ofdm = OFDMModulator(resource_grid=grid.grid, new_fft_size=fft_size)
tx_signal, sample_rate = ofdm.modulate()
tx_signal = tx_signal.astype(np.complex64)

# Centriranje TX signala
tx_signal -= np.mean(tx_signal)

# ================================================================
# 5. AWGN kanal
# ================================================================
awgn = AWGNChannel(snr_db=snr_db, seed=42)
rx_signal = awgn.apply(tx_signal)

# Centriranje RX signala
rx_signal -= np.mean(rx_signal)

plot_len = min(2000, len(tx_signal))

# ================================================================
# 6. PBCH konstelacija (TX + RX)
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(np.real(pbch_symbols), np.imag(pbch_symbols),
            s=40, label="PBCH TX")
plt.scatter(np.real(pbch_symbols_awgn), np.imag(pbch_symbols_awgn),
            s=40, label="PBCH RX (AWGN)")
plt.title("PBCH QPSK Konstelacija (AWGN)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "awgn_pbch_constellation_tx_rx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 7. OFDM konstelacija – TX
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(np.real(tx_signal[:plot_len]),
            np.imag(tx_signal[:plot_len]),
            s=5, label="OFDM TX")
plt.title("OFDM Konstelacija TX (prije AWGN)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "awgn_ofdm_constellation_tx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 8. OFDM konstelacija – RX
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(np.real(rx_signal[:plot_len]),
            np.imag(rx_signal[:plot_len]),
            s=5, label="OFDM RX")
plt.title("OFDM Konstelacija RX (poslije AWGN)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "awgn_ofdm_constellation_rx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 9. PBCH faza – TX
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.angle(pbch_symbols), label="PBCH TX")
plt.title("PBCH Faza TX (prije AWGN)")
plt.xlabel("Simbol")
plt.ylabel("Faza [rad]")
plt.grid(True)
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "awgn_pbch_phase_tx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 10. PBCH faza – RX
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.angle(pbch_symbols_awgn),
         label=f"PBCH RX ({snr_db} dB)")
plt.title("PBCH Faza RX (poslije AWGN)")
plt.xlabel("Simbol")
plt.ylabel("Faza [rad]")
plt.grid(True)
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "awgn_pbch_phase_rx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 11. OFDM faza – RX
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.angle(rx_signal[:plot_len]),
         label=f"OFDM RX ({snr_db} dB)")
plt.title("OFDM Faza RX (poslije AWGN)")
plt.xlabel("Uzorak")
plt.ylabel("Faza [rad]")
plt.grid(True)
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "awgn_ofdm_phase_rx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 12. OFDM realni dio – TX
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.real(tx_signal[:plot_len]), label="OFDM TX")
plt.title("OFDM Realni dio TX")
plt.xlabel("Uzorak")
plt.ylabel("Amplituda")
plt.grid(True)
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "awgn_ofdm_real_tx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 13. OFDM realni dio – RX
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.real(rx_signal[:plot_len]), label="OFDM RX")
plt.title("OFDM Realni dio RX")
plt.xlabel("Uzorak")
plt.ylabel("Amplituda")
plt.grid(True)
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "awgn_ofdm_real_rx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
