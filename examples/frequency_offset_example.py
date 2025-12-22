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
fft_size = 128
plot_len = 2000
delta_f_hz = 500  # frekvencijski offset [Hz]

# ================================================================
# 1. PSS signal (dummy)
# ================================================================
pss_signal = PSSGenerator.generate(1)

# Dummy QPSK simboli (za ilustraciju offseta)
qpsk_symbols = np.array([
    1 + 1j,
    -1 + 1j,
    -1 - 1j,
    1 - 1j
]) / np.sqrt(2)
qpsk_symbols = np.tile(qpsk_symbols, 50)

# ================================================================
# 2. Resource Grid
# ================================================================
grid = ResourceGrid(
    ndlrb=ndlrb,
    num_subframes=num_subframes,
    normal_cp=True
)
grid.map_pss(pss_sequence=pss_signal, symbol_index=pss_symbol_idx)

# ================================================================
# 3. OFDM modulacija
# ================================================================
ofdm = OFDMModulator(resource_grid=grid.grid, new_fft_size=fft_size)
tx_signal, sample_rate = ofdm.modulate()
tx_signal = tx_signal.astype(np.complex64)

# ================================================================
# 4. Primjena Frequency Offset
# ================================================================
freq_offset = FrequencyOffset(
    freq_offset_hz=delta_f_hz,
    sample_rate_hz=sample_rate
)
rx_signal_ofdm = freq_offset.apply(tx_signal)

# QPSK: simulacija frekvencijskog offseta
t = np.arange(len(qpsk_symbols)) / sample_rate
qpsk_symbols_rx = qpsk_symbols * np.exp(1j * 2 * np.pi * delta_f_hz * t)

# ================================================================
# 5. QPSK konstelacije (TX / RX)
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(np.real(qpsk_symbols), np.imag(qpsk_symbols),
            s=40, label="QPSK TX")
plt.title("QPSK Konstelacija TX (bez offseta)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "offset_qpsk_constellation_tx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(np.real(qpsk_symbols_rx), np.imag(qpsk_symbols_rx),
            s=40, label="QPSK RX + Offset")
plt.title("QPSK Konstelacija RX (frekvencijski offset)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "offset_qpsk_constellation_rx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 6. OFDM konstelacija – TX
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(np.real(tx_signal[:plot_len]),
            np.imag(tx_signal[:plot_len]),
            s=5, label="OFDM TX")
plt.title("OFDM Konstelacija TX (bez offseta)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "offset_ofdm_constellation_tx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 7. OFDM konstelacija – RX (sa offsetom)
# ================================================================
plt.figure(figsize=(6, 6))
plt.scatter(np.real(rx_signal_ofdm[:plot_len]),
            np.imag(rx_signal_ofdm[:plot_len]),
            s=5, label="OFDM RX + Offset")
plt.title("OFDM Konstelacija RX (frekvencijski offset)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "offset_ofdm_constellation_rx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 8. OFDM faza – TX
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.angle(tx_signal[:plot_len]), label="OFDM TX")
plt.title("OFDM Faza TX")
plt.xlabel("Uzorak")
plt.ylabel("Faza [rad]")
plt.grid(True)
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "offset_ofdm_phase_tx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()

# ================================================================
# 9. OFDM faza – RX (sa offsetom)
# ================================================================
plt.figure(figsize=(10, 4))
plt.plot(np.angle(rx_signal_ofdm[:plot_len]),
         label="OFDM RX + Offset")
plt.title("OFDM Faza RX (frekvencijski offset)")
plt.xlabel("Uzorak")
plt.ylabel("Faza [rad]")
plt.grid(True)
plt.legend()
plt.savefig(
    os.path.join(RESULTS_DIR, "offset_ofdm_phase_rx.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
