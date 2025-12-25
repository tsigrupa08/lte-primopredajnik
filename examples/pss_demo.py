"""
pss_demo.py
===========

Primjer generisanja i vizualizacije LTE Primary Synchronization Signal (PSS)
sekvenci za N_ID_2 = 0, 1 i 2.

TX-only primjer (bez kanala i prijemnika).
"""
from __future__ import annotations

import os
import sys

# Dodaj root projekta u PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import numpy as np
import matplotlib.pyplot as plt

from transmitter.pss import generate_pss_sequence

# ================================================================
# Results folders (TX-only)
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(BASE_DIR, "results")
TX_DIR = os.path.join(RESULTS_BASE, "tx")

os.makedirs(TX_DIR, exist_ok=True)


def main() -> None:
    """
    Generiše i vizualizira sve tri LTE PSS sekvence (N_ID_2 = 0, 1, 2):
    - magnituda |d[n]|
    - faza ∠d[n]
    """

    nid2_values = [0, 1, 2]
    pss_sequences = []

    for nid2 in nid2_values:
        seq = generate_pss_sequence(nid2)
        if seq.shape[0] != 62:
            raise ValueError(
                f"Očekivana dužina PSS sekvence je 62, a dobijeno {seq.shape[0]}"
            )
        pss_sequences.append(seq)

    n = np.arange(62)

    # --------------------------------------------------
    # 1) Magnituda |d[n]|
    # --------------------------------------------------
    fig_mag, axes_mag = plt.subplots(
        len(nid2_values), 1, figsize=(8, 6),
        sharex=True, constrained_layout=True
    )

    for ax, seq, nid2 in zip(axes_mag, pss_sequences, nid2_values):
        ax.stem(n, np.abs(seq))
        ax.set_ylabel("|d[n]|")
        ax.set_title(f"PSS magnituda (N_ID_2 = {nid2})")
        ax.grid(True, linestyle=":")

    axes_mag[-1].set_xlabel("n (indeks uzorka)")
    fig_mag.suptitle("Magnitude LTE PSS sekvenci", fontsize=14)

    fig_mag.savefig(
        os.path.join(TX_DIR, "pss_magnitude.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_mag)

    # --------------------------------------------------
    # 2) Faza ∠d[n]
    # --------------------------------------------------
    fig_phase, axes_phase = plt.subplots(
        len(nid2_values), 1, figsize=(8, 6),
        sharex=True, constrained_layout=True
    )

    for ax, seq, nid2 in zip(axes_phase, pss_sequences, nid2_values):
        ax.stem(n, np.angle(seq))
        ax.set_ylabel("ϕ[n] (rad)")
        ax.set_title(f"PSS faza (N_ID_2 = {nid2})")
        ax.grid(True, linestyle=":")

    axes_phase[-1].set_xlabel("n (indeks uzorka)")
    fig_phase.suptitle("Faze LTE PSS sekvenci", fontsize=14)

    fig_phase.savefig(
        os.path.join(TX_DIR, "pss_phase.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_phase)

    print("[OK] PSS demo završen.")
    print("Rezultati su snimljeni u examples/results/tx/")


if __name__ == "__main__":
    main()
