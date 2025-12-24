import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from transmitter.pss import PSSGenerator
from transmitter.resource_grid import ResourceGrid
from transmitter.ofdm import OFDMModulator
from channel.frequency_offset import FrequencyOffset

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
fft_size = 128
plot_len = 2000
delta_f_hz = 500  # frekvencijski offset [Hz]

# ================================================================
# 1. PSS signal (TX)
# ================================================================
pss_signal = PSSGenerator.generate(1)

# ================================================================
# 2. QPSK simboli (TX – ilustracija)
# ================================================================
qpsk_symbols = np.array([
    1 + 1j,
    -1 + 1j,
    -1 - 1j,
    1 - 1j
]) / np.sqrt(2)
qpsk_symbols = np.tile(qpsk_symbols, 50)

# ================================================================
# 3. Resource grid (TX)
# ================================================================
grid = ResourceGrid(
    ndlrb=ndlrb,
    num_subframes=num_subframes,
    normal_cp=True
)
grid.map_pss(pss_sequence=pss_signal, symbol_index=pss_symbol_idx)

# ================================================================
# 4. OFDM modulacija (TX)
# ================================================================
ofdm = OFDMModulator(resource_grid=grid.grid, new_fft_size=fft_size)
tx_signal, fs = ofdm.modulate()
tx_signal = tx_signal.astype(np.complex64)

# ================================================================
# 5. Frequency offset kanal (CHANNEL)
# ================================================================
freq_offset = FrequencyOffset(
    freq_offset_hz=delta_f_hz,
    sample_rate_hz=fs
)
rx_signal = freq_offset.apply(tx_signal)

# QPSK – čisti efekat frekvencijskog offseta
t = np.arange(len(qpsk_symbols)) / fs
qpsk_symbols_rx = qpsk_symbols * np.exp(1j * 2 * np.pi * delta_f_hz * t)

# ================================================================
# TX: QPSK konstelacija
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(qpsk_symbols.real, qpsk_symbols.imag, s=40)
plt.title("QPSK konstelacija (TX)")
plt.grid(True)
plt.axis("equal")
plt.savefig(
    os.path.join(TX_DIR, "offset_qpsk_constellation_tx.png"),
    dpi=300
)
plt.close()

# ================================================================
# CHANNEL: QPSK konstelacija (efekat offseta)
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(qpsk_symbols_rx.real, qpsk_symbols_rx.imag, s=40)
plt.title("QPSK konstelacija (frequency offset)")
plt.grid(True)
plt.axis("equal")
plt.savefig(
    os.path.join(CH_DIR, "offset_qpsk_constellation_channel.png"),
    dpi=300
)
plt.close()

# ================================================================
# RX: QPSK konstelacija
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(qpsk_symbols_rx.real, qpsk_symbols_rx.imag, s=40)
plt.title("QPSK konstelacija (RX)")
plt.grid(True)
plt.axis("equal")
plt.savefig(
    os.path.join(RX_DIR, "offset_qpsk_constellation_rx.png"),
    dpi=300
)
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
plt.savefig(
    os.path.join(TX_DIR, "offset_ofdm_constellation_tx.png"),
    dpi=300
)
plt.close()

# ================================================================
# RX: OFDM konstelacija
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(rx_signal[:plot_len].real,
            rx_signal[:plot_len].imag, s=5)
plt.title("OFDM konstelacija (RX – frequency offset)")
plt.grid(True)
plt.axis("equal")
plt.savefig(
    os.path.join(RX_DIR, "offset_ofdm_constellation_rx.png"),
    dpi=300
)
plt.close()

# ================================================================
# TX: OFDM faza
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.angle(tx_signal[:plot_len]))
plt.title("OFDM faza (TX)")
plt.grid(True)
plt.savefig(
    os.path.join(TX_DIR, "offset_ofdm_phase_tx.png"),
    dpi=300
)
plt.close()

# ================================================================
# RX: OFDM faza
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.angle(rx_signal[:plot_len]))
plt.title("OFDM faza (RX – frequency offset)")
plt.grid(True)
plt.savefig(
    os.path.join(RX_DIR, "offset_ofdm_phase_rx.png"),
    dpi=300
)
plt.close()

print("[OK] Frequency offset example završen.")
print("Rezultati su snimljeni u examples/results/{tx,channel,rx}/")
