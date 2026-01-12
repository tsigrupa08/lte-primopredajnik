"""
LTE PBCH End-to-End Simulacija Performansi (AWGN)

Napomena:
- U ovoj implementaciji PBCH 960 QPSK simbola mapira se kao 4 bloka po 240
  kroz 4 subfrejma (pojednostavljeno u odnosu na realni LTE raspored).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) FIX: omogućava import transmitter/receiver kad pokrećeš iz examples/
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- IMPORTI MODULA ---
from transmitter.LTETxChain import LTETxChain
from channel.awgn_channel import AWGNChannel
from receiver.LTERxChain import LTERxChain


def _get_rx_outputs(rx_result):
    """
    Podržava i:
      - RxResult dataclass (rx_result.mib_bits, rx_result.crc_ok)
      - dict (rx_result['mib_bits'], rx_result['crc_ok'])
    """
    if hasattr(rx_result, "mib_bits") and hasattr(rx_result, "crc_ok"):
        return rx_result.mib_bits, bool(rx_result.crc_ok)
    if isinstance(rx_result, dict):
        return rx_result.get("mib_bits", None), bool(rx_result.get("crc_ok", False))
    return None, False


def run_simulation():
    # 1) KONFIGURACIJA SIMULACIJE
    snr_points = [0, 5, 10, 15, 20, 25]
    num_trials = 100

    NDLRB = 6
    NORMAL_CP = True
    NUM_SUBFRAMES = 4   # OBAVEZNO >= 4 u tvojoj TX implementaciji (4×240=960)

    ber_results = []
    crc_success_results = []

    print(f"Pokrećem simulaciju: {len(snr_points)} SNR tačaka, {num_trials} trial-a po tački.")
    print("-" * 85)
    print(f"{'SNR [dB]':<10} | {'BER':<12} | {'CRC Success':<12} | {'Errors/Total Bits':<25}")
    print("-" * 85)

    # TX chain može biti jedan (generate_waveform resetuje grid interno)
    tx_chain = LTETxChain(
        n_id_2=0,
        ndlrb=NDLRB,
        num_subframes=NUM_SUBFRAMES,
        normal_cp=NORMAL_CP
    )

    for snr in snr_points:
        crc_ok_count = 0
        bit_err = 0
        total_bits = 0

        channel = AWGNChannel(snr_db=snr, seed=None)

        # Napravi RX chain jednom po SNR (brže, nema state koji smeta)
        # Neki RX chainovi traže sample_rate_hz, neki ne → try/except
        rx_chain = None

        for trial in range(num_trials):
            # A) TX: 24-bit MIB
            mib_tx = np.random.randint(0, 2, 24, dtype=np.uint8)

            tx_waveform, fs = tx_chain.generate_waveform(mib_bits=mib_tx)

            # lazy init RX (znamo fs tek nakon TX)
            if rx_chain is None:
                try:
                    rx_chain = LTERxChain(sample_rate_hz=fs, ndlrb=NDLRB, normal_cp=NORMAL_CP, pci=0)
                except TypeError:
                    rx_chain = LTERxChain(ndlrb=NDLRB, normal_cp=NORMAL_CP, pci=0)

            # B) KANAL: AWGN
            rx_waveform = channel.apply(tx_waveform)

            # C) RX: decode/process
            if hasattr(rx_chain, "decode"):
                result = rx_chain.decode(rx_waveform)
            else:
                result = rx_chain.process(rx_waveform)  # fallback ako neko ipak ima process()

            mib_rx, is_crc_ok = _get_rx_outputs(result)

            # D) METRIKE
            if is_crc_ok:
                crc_ok_count += 1

            # BER: ako nema izlaza (None) ili dimenzija nije 24 → broj sve kao grešku
            L_tx = 24
            if mib_rx is None or len(mib_rx) != L_tx:
                bit_err += L_tx
                total_bits += L_tx
            else:
                mib_rx = np.asarray(mib_rx, dtype=np.uint8)
                errors = int(np.count_nonzero(mib_tx != mib_rx))
                bit_err += errors
                total_bits += L_tx

        ber = bit_err / total_bits if total_bits > 0 else 1.0
        crc_rate = crc_ok_count / num_trials

        ber_results.append(ber)
        crc_success_results.append(crc_rate)

        print(f"{snr:<10.1f} | {ber:<12.6f} | {crc_rate:<12.2%} | {bit_err}/{total_bits:<25d}")

    print("-" * 85)
    print("Simulacija završena.")

    plot_combined_results(snr_points, ber_results, crc_success_results, num_trials=num_trials)


def plot_combined_results(snr_points, ber, crc_success, num_trials=100):
    snr_points = np.array(snr_points, dtype=float)
    ber = np.array(ber, dtype=float)
    crc_success = np.array(crc_success, dtype=float)

    # FIX: semilogy ne voli nulu → postavi min na 1/(ukupno bitova)
    min_ber = 1.0 / (max(num_trials, 1) * 24.0)
    ber_plot = np.clip(ber, min_ber, 1.0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("SNR [dB]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    line1 = ax1.semilogy(snr_points, ber_plot, "o--", linewidth=2, markersize=8, label="BER")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("CRC Success Rate (0–1)", fontsize=12, fontweight="bold")
    line2 = ax2.plot(snr_points, crc_success, "s-", linewidth=2, markersize=8, label="CRC Success")
    ax2.set_ylim(-0.05, 1.05)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)

    plt.title("End-to-End LTE PBCH Performanse: BER i CRC Success vs SNR", fontsize=14, y=1.15)
    plt.tight_layout()

    output_dir = os.path.join("examples", "results", "rx")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "ber_crc_vs_snr.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nGrafik sačuvan: {output_path}")

    plt.show()


if __name__ == "__main__":
    run_simulation()
