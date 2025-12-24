"""
grid_demo.py
============

Demonstracija mapiranja PSS i PBCH simbola u LTE resource grid.

Ovaj skript koristi klasu :class:`transmitter.resource_grid.ResourceGrid`
da kreira LTE resource grid za NDLRB = 6 (1.4 MHz), zatim:

* generiše demo PSS sekvencu dužine 62 (amplituda 1),
* mapira PSS u OFDM simbol l = 6,
* generiše demo PBCH QPSK simbole (npr. 240 simbola),
* mapira PBCH u simbole l = 7, 8, 9, 10,
* prikazuje u jednom prozoru:
    1. 2D heatmap magnitude resource grida sa jasno označenim
       PSS, PBCH i DC podnosiocem + legendom boja,
    2. 1D graf amplitude PSS-a u simbolu l = 6.

Kako pokrenuti
--------------

Iz root direktorija projekta (gdje je folder ``examples/``):

PowerShell / CMD:

>>> python -m examples.grid_demo

ili:

>>> python examples/grid_demo.py
"""

from __future__ import annotations

from typing import Iterable, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transmitter.resource_grid import ResourceGrid


# -----------------------------------------------------------------------------
# PUTANJA ZA FIGURE (JEDINA FUNKCIONALNA IZMJENA)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# KREIRANJE DEMO GRIDA (PSS + PBCH)
# -----------------------------------------------------------------------------
def create_demo_grid(
    ndlrb: int = 6,
    num_subframes: int = 1,
    normal_cp: bool = True,
) -> Tuple[ResourceGrid, np.ndarray, np.ndarray, Iterable[int]]:
    """
    Kreira demo LTE resource grid i mapira PSS + PBCH.

    Parametri
    ----------
    ndlrb : int, opcionalno
        Broj downlink resource blokova. Za LTE 1.4 MHz vrijedi 6.
    num_subframes : int, opcionalno
        Broj subfrejmova u gridu. U demou tipično 1.
    normal_cp : bool, opcionalno
        Ako je ``True``, koristi se normalni CP (14 simbola po subfrejmu),
        u suprotnom prošireni CP (12 simbola po subfrejmu).

    Povratna vrijednost
    -------------------
    rg : ResourceGrid
        Objekat koji sadrži popunjeni LTE resource grid.
    pss_sequence : ndarray kompleksnih brojeva
        PSS sekvenca dužine 62 koja je mapirana u grid.
    pbch_symbols : ndarray kompleksnih brojeva
        PBCH QPSK simboli koji su mapirani u grid.
    pbch_symbol_indices : iterable int
        Indeksi OFDM simbola u koje je PBCH mapiran.

    Primjeri
    --------
    >>> from examples.grid_demo import create_demo_grid
    >>> rg, pss, pbch, l_pbch = create_demo_grid()
    >>> rg.grid.shape
    (72, 14)
    """
    rg = ResourceGrid(ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=normal_cp)

    n_pss = 62
    pss_sequence = np.ones(n_pss, dtype=complex)
    pss_symbol_index = 6
    rg.map_pss(pss_sequence=pss_sequence, symbol_index=pss_symbol_index)

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
    rg.map_pbch(
        pbch_symbols=pbch_symbols,
        pbch_symbol_indices=pbch_symbol_indices,
        reserved_re_mask=None,
    )

    return rg, pss_sequence, pbch_symbols, pbch_symbol_indices


# -----------------------------------------------------------------------------
# CRTANJE REZULTATA
# -----------------------------------------------------------------------------
def plot_results(
    rg: ResourceGrid,
    pss_symbol_index: int = 6,
    pbch_symbol_indices: Iterable[int] = (7, 8, 9, 10),
    save_path: str | None = None,
) -> None:
    """
    Prikazuje rezultate mapiranja PSS + PBCH u jednom prozoru.
    """
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "grid_demo_result.png")

    grid_abs = np.abs(rg.grid)
    num_subcarriers, _ = grid_abs.shape
    k = np.arange(num_subcarriers)
    pbch_symbol_indices = list(pbch_symbol_indices)

    dc_index = (rg.ndlrb * 12) // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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

    pss_line = ax1.axvline(pss_symbol_index, color="red", linestyle="--", linewidth=1.2)
    lmin, lmax = min(pbch_symbol_indices), max(pbch_symbol_indices)
    pbch_span = ax1.axvspan(lmin - 0.5, lmax + 0.5, alpha=0.15, color="white")
    dc_line = ax1.axhline(dc_index, color="black", linestyle=":")

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("|RE|")

    low_patch = mpatches.Patch(color="#440154", label="Prazno RE (|RE| ≈ 0)")
    high_patch = mpatches.Patch(color="#fde725", label="Mapirani simboli (|RE| ≈ 1)")

    ax1.legend(
        handles=[pss_line, pbch_span, dc_line, low_patch, high_patch],
        loc="lower left",
        fontsize=7,
        framealpha=0.8,
    )

    column = grid_abs[:, pss_symbol_index]
    ax2.stem(k, column)
    ax2.set_xlabel("Subcarrier indeks (k)")
    ax2.set_ylabel(f"|RE(k, l = {pss_symbol_index})|")
    ax2.set_title(f"PSS u simbolu l = {pss_symbol_index}")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Glavna funkcija za demonstraciju mapiranja PSS + PBCH.
    """
    rg, pss_sequence, pbch_symbols, pbch_symbol_indices = create_demo_grid()

    print("=== LTE resource grid demo ===")
    print(f"NDLRB: {rg.ndlrb}")
    print(f"Broj podnosioca: {rg.num_subcarriers}")
    print(f"Broj OFDM simbola ukupno: {rg.num_symbols_total}")
    print(f"PSS dužina: {pss_sequence.shape[0]}")
    print(f"PBCH broj simbola: {pbch_symbols.shape[0]}")
    print(f"PBCH simboli mapirani u l = {pbch_symbol_indices}")

    plot_results(rg, pss_symbol_index=6, pbch_symbol_indices=pbch_symbol_indices)

    print("Slika je snimljena u 'examples/figures/grid_demo_result.png'.")


if __name__ == "__main__":
    main()
