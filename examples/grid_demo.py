"""
grid_demo.py
============

Demonstracija mapiranja PSS i PBCH simbola u LTE resource grid.
"""
from __future__ import annotations

import sys
import os

# dodaj root projekta u PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transmitter.resource_grid import ResourceGrid



# ================================================================
# Results folder: examples/results
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# KREIRANJE DEMO GRIDA (PSS + PBCH)
# -----------------------------------------------------------------------------
def create_demo_grid(
    ndlrb: int = 6,
    num_subframes: int = 1,
    normal_cp: bool = True,
) -> Tuple[ResourceGrid, np.ndarray, np.ndarray, Iterable[int]]:

    rg = ResourceGrid(ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=normal_cp)

    # PSS demo – konstanta amplitude 1
    n_pss = 62
    pss_sequence = np.ones(n_pss, dtype=complex)
    pss_symbol_index = 6
    rg.map_pss(pss_sequence=pss_sequence, symbol_index=pss_symbol_index)

    # PBCH demo – QPSK
    num_pbch_symbols = 240
    bits = np.random.randint(0, 2, size=(2 * num_pbch_symbols,))
    bits = bits.reshape(-1, 2)

    mapping = {
        (0, 0): (1 + 1j) / np.sqrt(2),
        (0, 1): (-1 + 1j) / np.sqrt(2),
        (1, 1): (-1 - 1j) / np.sqrt(2),
        (1, 0): (1 - 1j) / np.sqrt(2),
    }
    pbch_symbols = np.array([mapping[tuple(b)] for b in bits], dtype=complex)

    pbch_symbol_indices = [7, 8, 9, 10]
    rg.map_pbch(pbch_symbols=pbch_symbols, pbch_symbol_indices=pbch_symbol_indices)

    return rg, pss_sequence, pbch_symbols, pbch_symbol_indices


# -----------------------------------------------------------------------------
# CRTANJE REZULTATA
# -----------------------------------------------------------------------------
def plot_results(
    rg: ResourceGrid,
    pss_symbol_index: int = 6,
    pbch_symbol_indices: Iterable[int] = (7, 8, 9, 10),
) -> None:

    grid_abs = np.abs(rg.grid)
    num_subcarriers, _ = grid_abs.shape
    k = np.arange(num_subcarriers)
    pbch_symbol_indices = list(pbch_symbol_indices)

    dc_index = (rg.ndlrb * 12) // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ---------------- Heatmap ----------------
    im = ax1.imshow(
        grid_abs,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )

    ax1.set_xlabel("OFDM simbol indeks (l)")
    ax1.set_ylabel("Subcarrier indeks (k)")
    ax1.set_title("LTE Resource Grid – magnituda |RE|")

    pss_line = ax1.axvline(pss_symbol_index, color="red", linestyle="--",
                           linewidth=1.2, label="PSS")

    lmin, lmax = min(pbch_symbol_indices), max(pbch_symbol_indices)
    pbch_span = ax1.axvspan(lmin - 0.5, lmax + 0.5, alpha=0.15,
                            color="white", label="PBCH")

    dc_line = ax1.axhline(dc_index, color="black", linestyle=":",
                          linewidth=1.0, label="DC")

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("|RE|")

    low_patch = mpatches.Patch(color="#440154", label="Prazno RE")
    high_patch = mpatches.Patch(color="#fde725", label="Mapirani simboli")

    ax1.legend(
        handles=[pss_line, pbch_span, dc_line, low_patch, high_patch],
        loc="lower left",
        fontsize=7,
        framealpha=0.8,
    )

    # ---------------- PSS 1D ----------------
    column = grid_abs[:, pss_symbol_index]
    ax2.stem(k, column)
    ax2.set_xlabel("Subcarrier indeks (k)")
    ax2.set_ylabel("|RE|")
    ax2.set_title(f"PSS u simbolu l = {pss_symbol_index}")
    ax2.grid(True)

    fig.tight_layout()

    # ------------------------------------------------------------
    # SAVE FIGURE
    # ------------------------------------------------------------
    fig.savefig(
        os.path.join(RESULTS_DIR, "grid_resource_grid_pss_pbch.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:

    rg, pss_sequence, pbch_symbols, pbch_symbol_indices = create_demo_grid()

    print("=== LTE resource grid demo ===")
    print(f"NDLRB: {rg.ndlrb}")
    print(f"Broj podnosioca: {rg.num_subcarriers}")
    print(f"Ukupno OFDM simbola: {rg.num_symbols_total}")
    print(f"PSS dužina: {pss_sequence.shape[0]}")
    print(f"PBCH simboli mapirani u l = {pbch_symbol_indices}")

    plot_results(rg, pss_symbol_index=6, pbch_symbol_indices=pbch_symbol_indices)

    print("Slika je snimljena u 'examples/results/grid_resource_grid_pss_pbch.png'.")


if __name__ == "__main__":
    main()
