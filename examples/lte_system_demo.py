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


import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transmitter.LTETxChain import LTETxChain
from receiver.LTERxChain import LTERxChain
from receiver.pss_sync import PSSSynchronizer
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

FS_HZ = 1.92e6

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


def normalize_rms(x: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    x = np.asarray(x)
    r = np.sqrt(np.mean(np.abs(x) ** 2))
    if r < 1e-15:
        return x
    return x * (target_rms / r)


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
        freq_offset_hz=float(cfo_hz),
        sample_rate_hz=float(FS_HZ),
        snr_db=float(snr_db)
    )

    # -----------------------------------------------------
    # RX
    # -----------------------------------------------------
    rx = LTERxChain(
        sample_rate_hz=float(FS_HZ),
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

    tx_waveform = np.asarray(results["tx_waveform"])
    rx_waveform = np.asarray(results["rx_waveform"])
    dbg = results.get("debug", {}) or {}

    print(f"TX N_ID_2        : {TX_NID2}")
    print(f"Detected N_ID_2  : {results.get('detected_nid', None)}")
    print(f"Timing τ̂        : {results.get('tau_hat', None)}")

    cfoh = results.get("cfo_hat_hz", None)
    if cfoh is None:
        print("CFO_hat (Hz)     : None")
    else:
        print(f"CFO_hat (Hz)     : {float(cfoh):.1f}")

    print(f"CRC OK           : {results.get('crc_ok', False)}")

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
    # (ne oslanjamo se na debug ključeve, nego ponovo izračunamo korelaciju)
    # =====================================================
    tau_hat = results.get("tau_hat", None)
    if tau_hat is None:
        tau_hat = 0
    tau_hat = int(tau_hat)

    pss = PSSSynchronizer(sample_rate_hz=float(results["fs_hz"]), ndlrb=NDLRB, normal_cp=True)
    corr = pss.correlate(normalize_rms(rx_waveform, 1.0))
    abs_corr = np.abs(corr)

    if abs_corr.ndim == 1:
        abs_corr = abs_corr.reshape(1, -1)

    Ncorr = abs_corr.shape[1]
    start = max(tau_hat - PSS_ZOOM, 0)
    end = min(tau_hat + PSS_ZOOM, Ncorr)
    tau_local = tau_hat - start  # tačna pozicija τ̂ u lokalnom prozoru

    fig = plt.figure(figsize=(12, 4))
    offsets = [0, 1.2, 2.4]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    cands = getattr(pss, "n_id_2_candidates", [0, 1, 2])

    for i in range(min(3, abs_corr.shape[0])):
        c = abs_corr[i, start:end]
        c = c / (np.max(c) + 1e-12)
        y = c + offsets[i]
        lab = f"N_ID_2={cands[i] if i < len(cands) else i}"
        plt.plot(y, color=colors[i], linewidth=2, label=lab)

    plt.axvline(tau_local, color="red", linestyle="--", linewidth=2, label="τ̂")
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
# PANEL 4 – PBCH bitovi (MIB) + CRC status
# =====================================================
    rx_bits = results.get("mib_bits_rx", None)

    # Ako CRC fail, RX vraća None -> uzmi "mib_hat_24" iz debug-a da možeš vidjeti greške
    if rx_bits is None:
        rx_bits = dbg.get("mib_hat_24", None)
        rx_label = "RX bits (CRC FAIL, mib_hat_24)"
    else:
        rx_label = "RX bits (CRC OK)"

    fig = plt.figure(figsize=(12, 4))
    idx = np.arange(24)

    plt.step(idx, tx_bits[:24], where="post", linewidth=2, label="TX bits")

    if rx_bits is not None:
        rx_bits = np.asarray(rx_bits, dtype=np.uint8).flatten()[:24]
        plt.step(idx, rx_bits, where="post", linestyle="--", linewidth=2, label=rx_label)

        errors = (tx_bits[:24] != rx_bits)
        if np.any(errors):
            plt.plot(idx[errors], rx_bits[errors], "rx", label="Greška (mismatch)")
    else:
        plt.text(
            0.5, 0.5,
            "RX nije vratio ni mib_hat_24 (decode prekinut ranije).",
            transform=plt.gca().transAxes,
            ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9)
        )

    plt.title(f"Panel 4 – MIB bits | CRC OK = {results.get('crc_ok', False)} [{case_name}]")
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
