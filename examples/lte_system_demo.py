from __future__ import annotations

"""
LTE end-to-end demonstracija (Sprint 4)
=====================================

TX → Channel → RX  (preko klase LTESystem)

Vizualizacija (4 panela):
1) TX OFDM waveform (I/Q)
2) RX waveform nakon kanala (AWGN + CFO)
3) PSS korelacija (3 krive) + timing (τ̂)
4) PBCH bitovi (TX vs RX) + CRC status

Prikazuju se:
- REALNI slučaj  (AWGN + CFO)
- IDEALNI slučaj (bez šuma i bez CFO)

End-to-end: “od bitova do bitova”
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from transmitter.LTETxChain import LTETxChain
from receiver.LTERxChain import LTERxChain
from channel.lte_channel import LTEChannel
from LTE_system_.lte_system import LTESystem


# =========================================================
# PATH ZA SPREMANJE FIGURA
# =========================================================
SAVE_DIR = "examples/results/lte_system"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# GLOBALNI PARAMETRI
# =========================================================
NDLRB = 6
NUM_SUBFRAMES = 4
TX_NID2 = 1

N_SHOW = 2000      # broj uzoraka za prikaz waveforma
PSS_ZOOM = 300     # zoom oko detektovanog τ̂

# REALNI USLOVI
REAL_SNR_DB = 6
REAL_CFO_HZ = 2000

# IDEALNI USLOVI
IDEAL_SNR_DB = 100
IDEAL_CFO_HZ = 0


# =========================================================
# POMOĆNA FUNKCIJA – tekst box
# =========================================================
def add_box(fig, text: str):
    fig.text(
        0.70, 0.5, text,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.95),
        va="center",
        fontsize=10
    )


# =========================================================
# JEDAN E2E SCENARIJ
# =========================================================
def run_case(case_name: str, snr_db: float, cfo_hz: float):
    print(f"\n=== LTE END-TO-END DEMO ({case_name}) ===")

    # -----------------------------------------------------
    # TX
    # -----------------------------------------------------
    tx = LTETxChain(
        n_id_2=TX_NID2,
        ndlrb=NDLRB,
        num_subframes=NUM_SUBFRAMES,
        normal_cp=True
    )

    # -----------------------------------------------------
    # CHANNEL
    # -----------------------------------------------------
    channel = LTEChannel(
        freq_offset_hz=cfo_hz,
        sample_rate_hz=1.92e6,
        snr_db=snr_db
    )

    # -----------------------------------------------------
    # RX
    # -----------------------------------------------------
    rx = LTERxChain(
        sample_rate_hz=1.92e6,
        ndlrb=NDLRB,
        normal_cp=True
    )

    # -----------------------------------------------------
    # SISTEM (TX + Channel + RX)
    # -----------------------------------------------------
    system = LTESystem(tx=tx, ch=channel, rx=rx)

    # MIB bitovi
    tx_bits = np.random.randint(0, 2, 24, dtype=np.uint8)

    # Pokretanje end-to-end simulacije
    results = system.run(tx_bits)

    tx_waveform = results["tx_waveform"]
    rx_waveform = results["rx_waveform"]
    dbg = results["debug"]

    print(f"TX N_ID_2        : {TX_NID2}")
    print(f"Detected N_ID_2  : {results['detected_nid']}")
    print(f"Timing τ̂        : {results['tau_hat']}")
    print(f"CFO_hat (Hz)     : {results['cfo_hat_hz']:.1f}")
    print(f"CRC OK           : {results['crc_ok']}")

    # =====================================================
    # PANEL 1 – TX waveform
    # =====================================================
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax[0].plot(np.real(tx_waveform[:N_SHOW]), linewidth=1.2)
    ax[1].plot(np.imag(tx_waveform[:N_SHOW]), linewidth=1.2)

    ax[0].set_title(f"Panel 1 – TX waveform (I) [{case_name}]")
    ax[1].set_title(f"Panel 1 – TX waveform (Q) [{case_name}]")
    ax[1].set_xlabel("Uzorak")

    for a in ax:
        a.set_ylabel("Amplituda")
        a.grid(True)

    add_box(fig, "TX signal prije kanala.\nOFDM signal (referenca).")
    plt.tight_layout(rect=[0, 0, 0.68, 1])
    plt.savefig(f"{SAVE_DIR}/{case_name.lower()}_panel1_tx.png", dpi=150)
    plt.close()

    # =====================================================
    # PANEL 2 – RX waveform
    # =====================================================
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax[0].plot(np.real(rx_waveform[:N_SHOW]), linewidth=1.2)
    ax[1].plot(np.imag(rx_waveform[:N_SHOW]), linewidth=1.2)

    ax[0].set_title(f"Panel 2 – RX waveform (I) [{case_name}]")
    ax[1].set_title(f"Panel 2 – RX waveform (Q) [{case_name}]")
    ax[1].set_xlabel("Uzorak")

    for a in ax:
        a.set_ylabel("Amplituda")
        a.grid(True)

    add_box(fig, f"Nakon kanala:\nAWGN = {snr_db} dB\nCFO = {cfo_hz} Hz")
    plt.tight_layout(rect=[0, 0, 0.68, 1])
    plt.savefig(f"{SAVE_DIR}/{case_name.lower()}_panel2_rx.png", dpi=150)
    plt.close()

    # =====================================================
    # PANEL 3 – PSS korelacija + timing
    # =====================================================
    corr = dbg["pss_corr_metrics"]
    tau = int(results["tau_hat"])

    start = max(tau - PSS_ZOOM, 0)
    end = tau + PSS_ZOOM

    fig = plt.figure(figsize=(12, 4))
    offsets = [0, 1.2, 2.4]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i in range(3):
        c = np.abs(corr[i][start:end])
        c /= (np.max(c) + 1e-12)
        y = c + offsets[i]
        plt.plot(y, color=colors[i], linewidth=2, label=f"N_ID_2={i}")

    plt.axvline(PSS_ZOOM, color="red", linestyle="--", linewidth=2, label="τ̂")
    plt.yticks(offsets, ["N_ID_2=0", "N_ID_2=1", "N_ID_2=2"])
    plt.title(f"Panel 3 – PSS korelacija + timing [{case_name}]")
    plt.xlabel("Uzorak (lokalni prozor)")
    plt.ylabel("Norm. korelacija + offset")
    plt.grid(True)
    plt.legend()

    add_box(
        fig,
        "RX traži početak LTE okvira.\n"
        "Najveći pik → izabrani N_ID_2.\n"
        "Vertikalna linija označava τ̂."
    )

    plt.tight_layout(rect=[0, 0, 0.68, 1])
    plt.savefig(f"{SAVE_DIR}/{case_name.lower()}_panel3_pss.png", dpi=150)
    plt.close()

    # =====================================================
    # PANEL 4 – PBCH bitovi
    # =====================================================
    rx_bits = results["mib_bits_rx"]

    if rx_bits is not None:
        M = min(len(tx_bits), len(rx_bits))
        idx = np.arange(M)
        errors = tx_bits[:M] != rx_bits[:M]

        fig = plt.figure(figsize=(12, 4))
        plt.step(idx, tx_bits[:M], where="post", linewidth=2, label="TX bits")
        plt.step(idx, rx_bits[:M], where="post", linestyle="--", linewidth=2, label="RX bits")

        if np.any(errors):
            plt.plot(idx[errors], rx_bits[:M][errors], "rx", label="Greška")

        plt.title(f"Panel 4 – PBCH bits | CRC OK = {results['crc_ok']} [{case_name}]")
        plt.xlabel("Indeks bita")
        plt.ylabel("Vrijednost")
        plt.ylim([-0.2, 1.2])
        plt.grid(True)
        plt.legend()

        add_box(fig, "PBCH dekodiranje (MIB).")
        plt.tight_layout(rect=[0, 0, 0.68, 1])
        plt.savefig(f"{SAVE_DIR}/{case_name.lower()}_panel4_pbch.png", dpi=150)
        plt.close()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    np.random.seed(0)

    run_case("REAL", REAL_SNR_DB, REAL_CFO_HZ)
    run_case("IDEAL", IDEAL_SNR_DB, IDEAL_CFO_HZ)

    print("\nFigure snimljene u:", SAVE_DIR)
