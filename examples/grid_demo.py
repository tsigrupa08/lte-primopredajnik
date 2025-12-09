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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transmitter.resource_grid import ResourceGrid


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
    # 1) Prazan grid
    rg = ResourceGrid(ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=normal_cp)

    # 2) PSS demo – jednostavna sekvenca amplitude 1
    n_pss = 62
    pss_sequence = np.ones(n_pss, dtype=complex)
    pss_symbol_index = 6  # l = 6

    rg.map_pss(pss_sequence=pss_sequence, symbol_index=pss_symbol_index)

    # 3) PBCH demo – 240 QPSK simbola, |s| ≈ 1
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
    save_path: str | None = "examples/grid_demo_result.png",
) -> None:
    """
    Prikazuje rezultate mapiranja PSS + PBCH u jednom prozoru.

    Lijevi subplot:
        2D heatmap magnitude resource grida sa:
            * označenim PSS simbolom,
            * označenim PBCH opsegom,
            * označenim DC podnosiocem,
            * legendom boja koja objašnjava tamno ljubičastu i žutu.

    Desni subplot:
        1D graf amplitude PSS-a u simbolu ``l = pss_symbol_index``.

    Parametri
    ----------
    rg : ResourceGrid
        Objekat koji sadrži popunjeni resource grid.
    pss_symbol_index : int, opcionalno
        Indeks OFDM simbola u kojem je PSS mapiran.
    pbch_symbol_indices : iterable int, opcionalno
        Indeksi OFDM simbola u koje je PBCH mapiran.
    save_path : str ili None, opcionalno
        Ako nije ``None``, figura se snima na zadatu putanju (PNG).

    Povratna vrijednost
    -------------------
    None
    """
    grid_abs = np.abs(rg.grid)
    num_subcarriers, _ = grid_abs.shape
    k = np.arange(num_subcarriers)
    pbch_symbol_indices = list(pbch_symbol_indices)

    # DC podnosioc u indeksnom sistemu grida
    dc_index = (rg.ndlrb * 12) // 2  # npr. 6*12/2 = 36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ------------------ Lijevi subplot: heatmap ------------------
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

    # PSS vertikalna linija
    pss_line = ax1.axvline(
        pss_symbol_index,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"PSS (l = {pss_symbol_index})",
    )

    # PBCH opseg
    lmin, lmax = min(pbch_symbol_indices), max(pbch_symbol_indices)
    pbch_span = ax1.axvspan(
        lmin - 0.5,
        lmax + 0.5,
        alpha=0.15,
        color="white",
        label=f"PBCH (l = {lmin}–{lmax})",
    )

    # DC podnosioc – horizontalna linija
    dc_line = ax1.axhline(
        dc_index,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"DC subcarrier (k = {dc_index})",
    )

    # Strelice i tekstualne anotacije
    ax1.annotate(
        "PSS",
        xy=(pss_symbol_index, dc_index + 20),
        xytext=(pss_symbol_index + 0.8, dc_index + 25),
        arrowprops=dict(arrowstyle="->", color="red", linewidth=1.0),
        fontsize=8,
        color="red",
    )

    ax1.annotate(
        "PBCH",
        xy=(lmin + 1.5, 10),
        xytext=(lmin + 1.5, 20),
        arrowprops=dict(arrowstyle="->", color="white", linewidth=1.0),
        fontsize=8,
        color="white",
    )

    ax1.annotate(
        "DC subcarrier",
        xy=(1, dc_index),
        xytext=(1.5, dc_index + 10),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.0),
        fontsize=8,
        color="black",
    )

    # Colorbar + objašnjenje boja
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("|RE|")

    # ručna legenda za tamno ljubičastu i žutu (viridis colormap)
    low_patch = mpatches.Patch(color="#440154", label="Prazno RE (|RE| ≈ 0)")
    high_patch = mpatches.Patch(color="#fde725", label="Mapirani simboli (|RE| ≈ 1)")

    legend = ax1.legend(
    handles=[pss_line, pbch_span, dc_line, low_patch, high_patch],
    loc="lower left",      # donji lijevi ugao
    fontsize=7,            # manji font
    framealpha=0.8,        # malo providna pozadina
    borderpad=0.3,         # tanji okvir
    labelspacing=0.3,      # manji razmak između linija
    handlelength=1.5,
    handletextpad=0.6,
)

    # ------------------ Desni subplot: PSS 1D ------------------
    column = grid_abs[:, pss_symbol_index]
    ax2.stem(k, column)
    ax2.set_xlabel("Subcarrier indeks (k)")
    ax2.set_ylabel(f"|RE(k, l = {pss_symbol_index})|")
    ax2.set_title(f"PSS u simbolu l = {pss_symbol_index}")
    ax2.grid(True)

    fig.tight_layout()

    # Snimanje slike ako je traženo
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Glavna funkcija za demonstraciju mapiranja PSS + PBCH.

    1. Kreira demo LTE resource grid.
    2. Ispiše osnovne informacije o gridu.
    3. Prikazuje rezultate i snima sliku na disk.
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

    print("Slika je snimljena u 'examples/grid_demo_result.png'.")


if __name__ == "__main__":
    main()
