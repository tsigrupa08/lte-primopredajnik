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
# Dodajemo root projekta u sys.path da Python vidi tvoje foldere
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
    """
    Crta heatmapu RX grida i sprema je u examples/results/rx/.
    """

    
    results_dir = project_root / "examples" / "results" / "rx"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Određivanje imena fajla
    if out_path is None:
        out_file = results_dir / "rx_grid_heatmap.png"
    else:
        out_path = Path(out_path)
        # Ako je dato samo ime fajla, stavi ga u results/rx
        if out_path.parent == Path("."):
            out_file = results_dir / out_path.name
        else:
            
            out_file = out_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

    # Priprema podataka (Magnituda)
    mag = np.abs(rx_grid)
    n_subcarriers, n_symbols = mag.shape

    # Crtanje
    fig, ax = plt.subplots(figsize=(10, 6))

    # Heatmap (origin='lower' je bitan za frekvenciju)
    im = ax.imshow(
        mag, 
        aspect="auto", 
        origin="lower", 
        cmap="viridis", 
        interpolation="nearest"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|RX Grid| Magnitude")

    # 1. PSS Oznaka
    if highlight_pss and n_symbols > pss_symbol_index:
        ax.axvline(x=pss_symbol_index, color="red", linestyle="--", alpha=0.8)
        
        # PSS = centralna 62 nosioca
        center_sc = n_subcarriers // 2
        pss_bw = 62
        lower_sc = max(0, center_sc - pss_bw // 2)
        
        rect_pss = patches.Rectangle(
            (pss_symbol_index - 0.5, lower_sc), 1, pss_bw,
            linewidth=2, edgecolor="red", facecolor="none", label="PSS (Sync)"
        )
        ax.add_patch(rect_pss)

    # 2. PBCH Oznaka
    if highlight_pbch:
        valid_pbch = [s for s in pbch_symbol_indices if s < n_symbols]
        if valid_pbch:
            start = valid_pbch[0]
            width = len(valid_pbch)
            rect_pbch = patches.Rectangle(
                (start - 0.5, 0), width, n_subcarriers,
                linewidth=2, edgecolor="cyan", facecolor="none", hatch="//", label="PBCH"
            )
            ax.add_patch(rect_pbch)

    # Stil
    ax.set_title(title)
    ax.set_xlabel("OFDM Symbol Index (Time)")
    ax.set_ylabel("Subcarrier Index (Freq)")
    
    if highlight_pss or highlight_pbch:
        ax.legend(loc="upper right")

    if annotate:
        info_text = (
            f"RX Grid: {n_subcarriers}x{n_symbols}\n"
            f"PSS loc: Sym {pss_symbol_index}\n"
            f"PBCH loc: Sym {pbch_symbol_indices}"
        )
        ax.text(
            0.02, 0.98, info_text, transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            fontsize=9
        )

    # Spremanje
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(out_file.resolve())


# -----------------------------------------------------------------------
# 3. GLAVNI PROGRAM (PIPELINE)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Pokrećem C3 RX Pipeline ---")

    # LAZY IMPORTS (Da izbjegnemo kružne greške)
    try:
        from transmitter.LTETxChain import LTETxChain
        from channel.lte_channel import LTEChannel
        from receiver.pss_sync import PSSSynchronizer
        from receiver.OFDM_demodulator import OFDMDemodulator
        print("[INFO] Klase učitane.")
    except ImportError as e:
        print(f"\n[GREŠKA] Ne mogu učitati klase: {e}")
        sys.exit(1)

    snr_db = 25
    
    # 1. TX GENERISANJE
    print("1. [TX] Generisanje signala...")
   
    try:
        tx = LTETxChain()
    except TypeError:
        tx = LTETxChain(cell_id=42)

    try:
        tx_waveform, fs = tx.generate_waveform()
    except TypeError:
        tx_waveform, fs = tx.generate_waveform(cell_id=42)

    # 2. KANAL
    print(f"2. [Channel] Dodajem šum (SNR={snr_db}dB)...")
    chan = LTEChannel(freq_offset_hz=0.0, sample_rate_hz=fs, snr_db=snr_db, seed=123)
    rx_waveform = chan.apply(tx_waveform)
    
    # 3. SINHRONIZACIJA
    print("3. [RX] PSS Sinhronizacija...")
    pss = PSSSynchronizer(sample_rate_hz=fs)
    
    corr = pss.correlate(rx_waveform)
    tau_hat, nid = pss.estimate_timing(corr)
    
    print(f"   --> Detektovani Cell ID: {nid}")
    print(f"   --> Timing offset: {tau_hat}")
    
    if tau_hat < 0 or tau_hat >= len(rx_waveform) - 1000:
        print("   [WARN] Sync fail, koristim 0.")
        tau_hat = 0
        
    rx_aligned = rx_waveform[tau_hat:]

    # 4. OFDM DEMODULACIJA
    print("4. [RX] OFDM Demodulacija...")
    # klasa koristi 'ndlrb'
    ofdm_rx = OFDMDemodulator(ndlrb=6)
    
    # 1. Dobijemo sirovi grid (Simboli x FFT)
    grid_raw = ofdm_rx.demodulate(rx_aligned)
    
    # 2. Izdvojimo aktivne (Simboli x 72)
    grid_active = ofdm_rx.extract_active_subcarriers(grid_raw)
    
    # 3. Transponujemo za plot (72 x Simboli)
    rx_grid_final = grid_active.T
    
    print(f"   --> Grid shape za plot: {rx_grid_final.shape}")

    # 5. VIZUALIZACIJA
    print("5. [PLOT] Crtanje...")
    path = plot_rx_grid_heatmap(
        rx_grid_final,
        title=f"RX Grid (CellID={nid}, SNR={snr_db}dB)",
        highlight_pss=True,
        highlight_pbch=True
    )
    
    print("------------------------------------------------")
    print(f"Spremljeno u RX folder: {path}")
    print("------------------------------------------------")