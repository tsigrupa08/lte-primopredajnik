from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------------------------------------------------
# 1. KONFIGURACIJA PROJEKTA
# -----------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# -----------------------------------------------------------------------
# 2. FUNKCIJA ZA VIZUALIZACIJU
# -----------------------------------------------------------------------
def plot_rx_grid_heatmap(
    rx_grid: np.ndarray,
    out_path: Optional[Union[str, Path]] = None,
    *,
    highlight_pss: bool = True,
    highlight_pbch: bool = True,
    pss_symbol_index: int = 6,
    pbch_symbol_indices: List[int] = [7, 8, 9, 10],
    title: str = "RX Grid Magnitude Heatmap (|RX|)",
    dpi: int = 180,
    annotate: bool = True,
) -> str:

    results_dir = project_root / "examples" / "results" / "rx"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_file = (
        results_dir / "rx_grid_heatmap.png"
        if out_path is None
        else Path(out_path)
    )

    mag = np.abs(rx_grid)
    n_subcarriers, n_symbols = mag.shape

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        mag,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|RX Grid| Magnitude")

    # ---------------- PSS ----------------
    if highlight_pss and pss_symbol_index < n_symbols:
        ax.axvline(pss_symbol_index, color="red", linestyle="--", alpha=0.8)

        center_sc = n_subcarriers // 2
        pss_bw = 62
        lower_sc = center_sc - pss_bw // 2

        ax.add_patch(
            patches.Rectangle(
                (pss_symbol_index - 0.5, lower_sc),
                1,
                pss_bw,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                label="PSS (LTE)",
            )
        )

    # ---------------- PBCH ----------------
    if highlight_pbch:
        valid = [s for s in pbch_symbol_indices if s < n_symbols]
        if valid:
            ax.add_patch(
                patches.Rectangle(
                    (valid[0] - 0.5, 0),
                    len(valid),
                    n_subcarriers,
                    linewidth=2,
                    edgecolor="cyan",
                    facecolor="none",
                    hatch="//",
                    label="PBCH (LTE)",
                )
            )

    ax.set_title(title)
    ax.set_xlabel("OFDM Symbol Index (LTE Frame)")
    ax.set_ylabel("Subcarrier Index (Frequency)")
    ax.legend(loc="upper right")

    if annotate:
        ax.text(
            0.02,
            0.98,
            f"RX Grid: {n_subcarriers} × {n_symbols}\n"
            f"PSS: symbol {pss_symbol_index}\n"
            f"PBCH: symbols {pbch_symbol_indices}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(out_file.resolve())


# -----------------------------------------------------------------------
# 3. GLAVNI PROGRAM
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Pokrećem RX pipeline (STABLE) ---")

    from transmitter.LTETxChain import LTETxChain
    from channel.lte_channel import LTEChannel
    from receiver.pss_sync import PSSSynchronizer
    from receiver.OFDM_demodulator import OFDMDemodulator

    snr_db = 25

    # TX
    print("1. [TX] Generisanje signala...")
    tx = LTETxChain()
    tx_waveform, fs = tx.generate_waveform()

    # Channel
    print("2. [Channel] Dodajem šum...")
    rx_waveform = LTEChannel(
        freq_offset_hz=0.0,
        sample_rate_hz=fs,
        snr_db=snr_db,
        seed=123,
    ).apply(tx_waveform)

    # PSS sync
    print("3. [RX] PSS sinhronizacija...")
    pss = PSSSynchronizer(sample_rate_hz=fs)
    corr = pss.correlate(rx_waveform)
    tau_hat, nid = pss.estimate_timing(corr)

    print(f"   Cell ID: {nid}")
    print(f"   Timing offset: {tau_hat}")

    if tau_hat < 0:
        tau_hat = 0

    # OFDM
    ofdm = OFDMDemodulator(ndlrb=6)

    # ✅ ROBUST FRAME ALIGN (bez cp_len)
    frame_start = max(0, tau_hat - 6 * ofdm.fft_size)
    rx_aligned = rx_waveform[frame_start:]

    print("4. [RX] OFDM demodulacija...")
    grid_raw = ofdm.demodulate(rx_aligned)
    grid_active = ofdm.extract_active_subcarriers(grid_raw)

    rx_grid = np.fft.fftshift(grid_active.T, axes=0)

    print(f"   RX grid shape: {rx_grid.shape}")

    print("5. [PLOT] Crtanje...")
    path = plot_rx_grid_heatmap(
        rx_grid,
        title=f"RX Grid (LTE, SNR={snr_db} dB)",
        pss_symbol_index=6,
        pbch_symbol_indices=[7, 8, 9, 10],
    )

    print("------------------------------------------------")
    print(f"Spremljeno: {path}")
    print("------------------------------------------------")
