"""
Vizualizacija C6 – End-to-end BER / CRC success vs SNR

Ova skripta pokreće cijeli LTE lanac (TX + Channel + RX) za različite
vrijednosti SNR-a i mjeri:
    - Bit Error Rate (BER)
    - CRC success rate

Na kraju crta graf koji pokazuje da RX radi bolje kada SNR raste.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Import modula iz projekta ---
from receiver.utils import RxUtils
from receiver.OFDM_demodulator import OFDMDemodulator
from receiver.resource_grid_extractor import PBCHExtractor
from receiver.QPSK_demapiranje import QPSKDemapper
from receiver.de_rate_matching import DeRateMatcher
from receiver.viterbi_decoder import ViterbiDecoder
from receiver.crc_checker import CRCChecker
from receiver.pss_sync import PSSSynchronizer
from receiver.LTERxChain import LTERxChain

from transmitter.LTETxChain import LTETxChain
from channel.lte_channel import LTEChannel
from LTE_system import LTE_System

# --- Parametri simulacije ---
snr_db_range = np.arange(-5, 21, 2)   # sweep od -5 do 20 dB
n_trials = 200                        # broj trial-ova po SNR-u
mib_len = 24                          # dužina MIB payload-a (prilagodi prema TX) - vjv 24 staviti

# --- Kreiraj lance ---
tx = LTETxChain(ndlrb=6, normal_cp=True)
rx = LTERxChain(ndlrb=6, normal_cp=True)

# --- Rezultati ---
ber_results = []
crc_success_results = []

rng = np.random.default_rng(123)

def random_mib_bits(n_bits=64):
    return rng.integers(0, 2, n_bits)

def compute_bit_errors(a, b):
    L = min(len(a), len(b))
    return np.sum(a[:L] != b[:L]), L

# --- Glavna petlja ---
for snr_db in snr_db_range:
    ch = LTEChannel(awgn_snr_db=snr_db, frequency_offset_hz=0.0)
    system = LTESystem(tx=tx, ch=ch, rx=rx)

    total_errors = 0
    total_bits = 0
    crc_ok_count = 0

    for _ in range(n_trials):
        bits_tx = random_mib_bits(mib_len)
        res = system.run(bits_tx)
        bits_rx = res["decoded_bits"]
        crc_ok = res["crc_ok"]

        errs, L = compute_bit_errors(bits_tx, bits_rx)
        total_errors += errs
        total_bits += L
        crc_ok_count += int(crc_ok)

    ber = total_errors / max(1, total_bits)
    crc_rate = crc_ok_count / n_trials

    ber_results.append(ber)
    crc_success_results.append(crc_rate)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(8, 5))

color_ber = 'tab:red'
ax1.set_xlabel('SNR [dB]')
ax1.set_ylabel('BER', color=color_ber)
ax1.semilogy(snr_db_range, ber_results, 'o-', color=color_ber, label='BER')
ax1.tick_params(axis='y', labelcolor=color_ber)
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

ax2 = ax1.twinx()
color_crc = 'tab:blue'
ax2.set_ylabel('CRC success rate', color=color_crc)
ax2.plot(snr_db_range, crc_success_results, 's-', color=color_crc, label='CRC Success')
ax2.tick_params(axis='y', labelcolor=color_crc)
ax2.set_ylim(0.0, 1.05)

plt.title('LTE End-to-End: BER i CRC success vs SNR')
fig.tight_layout()
plt.show()

