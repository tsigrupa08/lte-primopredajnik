from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np
import matplotlib.pyplot as plt


def plot_pbch_constellation(
    pbch_symbols: np.ndarray,
    out_path: Optional[Union[str, Path]] = None,
    *,
    show_ideal_qpsk: bool = True,
    show_gray_labels: bool = False,
    annotate: bool = True,
    hide_zeros_for_plot: bool = False,
    title: str = "PBCH konstelacija (izvučeni simboli prije demapiranja)",
    dpi: int = 180,
    zoom: Literal["auto", "robust", "ideal"] = "ideal",
    robust_percentile: float = 99.5,
    show_outliers_in_inset: bool = True,
) -> str:
    """
    RX-only prikaz PBCH QPSK konstelacije.
    Slika se sprema u examples/figures/.
    """

    # ------------------------------------------------------------
    # Putanje
    # ------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    figures_dir = project_root / "examples" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        out_file = figures_dir / "pbch_constellation.png"
    else:
        out_path = Path(out_path)
        out_file = figures_dir / out_path.name if out_path.parent == Path(".") else out_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Ulazni simboli
    # ------------------------------------------------------------
    x = np.asarray(pbch_symbols, dtype=np.complex128).ravel()
    if hide_zeros_for_plot:
        x = x[x != 0.0 + 0.0j]

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(
        x.real,
        x.imag,
        s=12,
        alpha=0.55,
        linewidths=0.0,
        label=f"PBCH RX simboli (N={x.size})",
    )

    # Idealne QPSK tačke – PRAZNI KRUGOVI
    ideal = None
    if show_ideal_qpsk:
        ideal = np.array(
            [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j],
            dtype=np.complex128,
        ) / np.sqrt(2)

        ax.scatter(
            ideal.real,
            ideal.imag,
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            label="Idealne QPSK tačke",
        )

        if show_gray_labels:
            labels = ["00", "01", "11", "10"]
            offsets = [(8, 8), (8, -16), (-16, -16), (-16, 8)]
            for lab, pt, (ox, oy) in zip(labels, ideal, offsets):
                ax.annotate(
                    lab,
                    (pt.real, pt.imag),
                    xytext=(ox, oy),
                    textcoords="offset points",
                    ha="left" if ox > 0 else "right",
                    va="bottom" if oy > 0 else "top",
                )

    # ------------------------------------------------------------
    # Stil
    # ------------------------------------------------------------
    ax.set_title(title)
    ax.set_xlabel("I komponenta (realni dio)")
    ax.set_ylabel("Q komponenta (imaginarni dio)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")

    # ------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------
    if x.size == 0:
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

    elif zoom == "ideal":
        ax.set_xlim([-1.25, 1.25])
        ax.set_ylim([-1.25, 1.25])

    elif zoom == "auto":
        m = max(1.0, np.max(np.abs(np.r_[x.real, x.imag])))
        ax.set_xlim([-1.3 * m, 1.3 * m])
        ax.set_ylim([-1.3 * m, 1.3 * m])

    else:
        p = robust_percentile
        lo, hi = 100 - p, p
        xlo, xhi = np.percentile(x.real, [lo, hi])
        ylo, yhi = np.percentile(x.imag, [lo, hi])
        padx = 0.2 * (xhi - xlo)
        pady = 0.2 * (yhi - ylo)
        ax.set_xlim([xlo - padx, xhi + padx])
        ax.set_ylim([ylo - pady, yhi + pady])

    ax.legend(loc="best")

    # ------------------------------------------------------------
    # INFO BOX (NEMA PREKLAPANJA)
    # ------------------------------------------------------------
    if annotate:
        ax.text(
            0.02,
            0.98,
            "RX-only PBCH konstelacija\n"
            "• Zumirano oko idealnih tačaka\n"
            "• Idealne tačke = prazni krugovi",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.85),
        )

    # ------------------------------------------------------------
    # Spremanje
    # ------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(out_file.resolve())


# ======================================================================
# POKRETANJE IZ TERMINALA
# ======================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ideal = np.array(
        [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j],
        dtype=np.complex128,
    ) / np.sqrt(2)

    rx_syms = ideal[rng.integers(0, 4, size=3000)]
    rx_syms += 0.15 * (
        rng.standard_normal(rx_syms.size)
        + 1j * rng.standard_normal(rx_syms.size)
    )

    path = plot_pbch_constellation(rx_syms)
    print("Spremljeno:", path)
