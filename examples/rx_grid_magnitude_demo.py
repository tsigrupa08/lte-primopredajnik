"""
RX-only: Demodulirani grid magnitude (poslije FFT) + oznake za PSS i PBCH

- generiše TX waveform (da RX ima "input")
- provuče kroz AWGN (opcionalno)
- PSS korelacija -> poravnanje na start SF0
- OFDM demod (FFT + CP remove) -> rx_grid_full i rx_grid_active (72xNs)
- heatmap |grid| (dB) + jasne oznake PSS i PBCH (boje + overlay + strelice)

Pokretanje:
    python examples/rx_grid_magnitude_demo.py
    python examples/rx_grid_magnitude_demo.py --snr 20 --seed 1

Output:
    examples/results/rx/rx_grid_magnitude_heatmap.png

    
Napomene:
    SF==SUBFRAME
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# ------------------------------------------------------------
# FIX importa kad pokrećeš iz examples/
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transmitter.LTETxChain import LTETxChain
from channel.awgn_channel import AWGNChannel
from receiver.pss_sync import PSSSynchronizer
from receiver.OFDM_demodulator import OFDMDemodulator
from receiver.resource_grid_extractor import pbch_symbol_indices_for_subframes


def normalize_rms(x: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    x = np.asarray(x)
    r = np.sqrt(np.mean(np.abs(x) ** 2))
    if r < 1e-15:
        return x
    return x * (target_rms / r)


def offset_samples_to_pss_cp_start(ofdm_demod: OFDMDemodulator, normal_cp: bool) -> int:
    """
    Koliko uzoraka od starta subfrejma do CP-start PSS simbola u SF0.
    Normal CP: PSS u l=6 (zadnji simbol slota0) -> prolaziš simbole 0..5.
    Extended CP: PSS u l=5 (zadnji simbol slota0).
    """
    N = int(ofdm_demod.fft_size)
    cps = ofdm_demod.cp_lengths  # per-slot

    if normal_cp:
        cp0 = int(cps[0])
        cp1 = int(cps[1])
        return (N + cp0) + 5 * (N + cp1)
    else:
        cp = int(cps[0])
        return 5 * (N + cp)


def samples_per_subframe(ofdm_demod: OFDMDemodulator) -> int:
    N = int(ofdm_demod.fft_size)
    cps = ofdm_demod.cp_lengths
    slot = int(sum((N + int(cp)) for cp in cps))
    return 2 * slot


def find_pss_peak(
    pss_sync: PSSSynchronizer,
    rx: np.ndarray,
    tau_expected: int,
    fft_size: int,
    win: int | None,
):
    """
    Vraća: (tau_hat, n_id_2_hat, peak_value)
    """
    corr = pss_sync.correlate(rx)
    abs_corr = np.abs(corr)

    if abs_corr.ndim == 1:
        abs_corr = abs_corr.reshape(1, -1)

    if win is None:
        win = max(2 * int(fft_size), 128)

    lo = max(0, int(tau_expected) - int(win))
    hi = min(abs_corr.shape[1], int(tau_expected) + int(win) + 1)

    if hi <= lo:
        k_idx, tau_hat = np.unravel_index(int(np.argmax(abs_corr)), abs_corr.shape)
        n_id_2_hat = int(pss_sync.n_id_2_candidates[int(k_idx)])
        peak = float(abs_corr[int(k_idx), int(tau_hat)])
        return int(tau_hat), int(n_id_2_hat), peak

    cand = np.arange(lo, hi)
    sub = abs_corr[:, cand]
    k_idx, c_idx = np.unravel_index(int(np.argmax(sub)), sub.shape)

    tau_hat = int(cand[int(c_idx)])
    n_id_2_hat = int(pss_sync.n_id_2_candidates[int(k_idx)])
    peak = float(abs_corr[int(k_idx), int(tau_hat)])
    return tau_hat, n_id_2_hat, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=30.0, help="SNR u dB (AWGN)")
    ap.add_argument("--seed", type=int, default=0, help="Seed za RNG (reproducibilnost)")
    ap.add_argument("--num_subframes", type=int, default=4, help="Broj subfrejmova (PBCH 4x240=960)")
    ap.add_argument("--dynamic_range", type=float, default=45.0, help="Dinamički opseg heatmape u dB")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # -------------------------
    # Parametri (kao u tvom setup-u)
    # -------------------------
    NDLRB = 6
    NORMAL_CP = True
    NUM_SF = int(args.num_subframes)

    # -------------------------
    # TX -> waveform (samo da RX ima šta da demoduliše)
    # -------------------------
    tx = LTETxChain(n_id_2=0, ndlrb=NDLRB, num_subframes=NUM_SF, normal_cp=NORMAL_CP)
    mib_bits = np.random.randint(0, 2, 24, dtype=np.uint8)
    tx_waveform, fs = tx.generate_waveform(mib_bits=mib_bits)

    # -------------------------
    # Kanal (AWGN)
    # -------------------------
    ch = AWGNChannel(snr_db=float(args.snr), seed=None)
    rx = ch.apply(tx_waveform)

    # (opc.) normalizuj radi ljepše heatmape
    rx = normalize_rms(rx, target_rms=1.0)

    # -------------------------
    # RX: PSS sync + poravnanje na SF0 + OFDM demod (FFT)
    # -------------------------
    ofdm = OFDMDemodulator(ndlrb=NDLRB, normal_cp=NORMAL_CP)
    pss = PSSSynchronizer(sample_rate_hz=float(fs), ndlrb=NDLRB, normal_cp=NORMAL_CP)

    tau_expected = offset_samples_to_pss_cp_start(ofdm, NORMAL_CP)
    tau_hat, n_id_2_hat, peak = find_pss_peak(pss, rx, tau_expected, ofdm.fft_size, win=None)

    # CFO procjena/korekcija preko PSS (da heatmap bude stabilniji)
    cfo_hat = float(pss.estimate_cfo(rx, int(tau_hat), int(n_id_2_hat)))
    rx_corr = pss.apply_cfo_correction(rx, cfo_hat)

    # Align na start SF0
    start_sf0_raw = int(tau_hat) - int(tau_expected)
    spsf = samples_per_subframe(ofdm)
    start_sf0 = int(np.rint(start_sf0_raw / spsf) * spsf) if spsf > 0 else start_sf0_raw
    if start_sf0 < 0:
        start_sf0 = 0
    rx_aligned = rx_corr[start_sf0:]

    # OFDM demod: grid_full (NFFT x Ns), pa active grid (72 x Ns)
    grid_full = ofdm.demodulate(rx_aligned)
    grid_active = ofdm.extract_active_subcarriers(grid_full)

    # -------------------------
    # Heatmap |grid| u dB
    # -------------------------
    mag = np.abs(grid_active)
    mag_db = 20.0 * np.log10(mag + 1e-12)

    vmax = float(np.max(mag_db))
    vmin = vmax - float(args.dynamic_range)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(
        f"RX grid |FFT| (active 72 subcarriers)  |  SNR={args.snr:.1f} dB  |  "
        f"tau_hat={tau_hat}  |  N_ID_2_hat={n_id_2_hat}  |  CFO_hat={cfo_hat:.2f} Hz",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("OFDM simbol indeks (l)")
    ax.set_ylabel("Subcarrier indeks (0..71)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|grid| [dB]")

    # -------------------------
    # Oznake za PSS i PBCH (jasnije, boje + overlay + strelice)
    # -------------------------
    PSS_COLOR = "tab:red"
    PBCH_COLOR = "tab:green"

    # PSS: normal CP -> l=6; zauzima k=5..66 (62 RE)
    l_pss = 6 if NORMAL_CP else 5
    k_pss_start = 5
    k_pss_len = 62

    # PSS overlay (poluprovidno)
    pss_rect = Rectangle(
        (l_pss - 0.5, k_pss_start - 0.5),
        1.0,
        float(k_pss_len),
        facecolor=PSS_COLOR,
        edgecolor=PSS_COLOR,
        alpha=0.45,
        linewidth=2.5,
        zorder=5,
    )
    ax.add_patch(pss_rect)

    # PSS label + strelica
    ax.annotate(
        "PSS",
        xy=(l_pss, k_pss_start + 0.5 * k_pss_len),
        xytext=(l_pss + 3.0, k_pss_start + 0.5 * k_pss_len + 10),
        color=PSS_COLOR,
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PSS_COLOR, lw=2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PSS_COLOR, alpha=0.92),
        zorder=10,
    )

    # PBCH simboli (po tvojoj postavci): [7,8,9,10] po subfrejmu
    pbch_cols = pbch_symbol_indices_for_subframes(num_subframes=NUM_SF, normal_cp=NORMAL_CP, start_subframe=0)

    # PBCH overlay: oboji cijelu kolonu (svi subcarriers) za svaki PBCH simbol
    for l in pbch_cols:
        pbch_rect = Rectangle(
            (l - 0.5, -0.5),
            1.0,
            72.0,
            facecolor=PBCH_COLOR,
            edgecolor=PBCH_COLOR,
            alpha=0.45,
            linewidth=1.8,
            zorder=4,
        )
        ax.add_patch(pbch_rect)

    # Jedna PBCH oznaka sa strelicom na prvi PBCH simbol
    l_pbch0 = int(pbch_cols[0]) if len(pbch_cols) > 0 else 7
    ax.annotate(
        "PBCH (l=[7,8,9,10] po SF)",
        xy=(l_pbch0, 60),
        xytext=(l_pbch0 + 5.0, 68),
        color=PBCH_COLOR,
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PBCH_COLOR, lw=2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PBCH_COLOR, alpha=0.92),
        zorder=10,
    )

    # Legenda (čista i jasna)
    legend_handles = [
        Patch(facecolor=PSS_COLOR, edgecolor=PSS_COLOR, alpha=0.25, label="PSS (l=6, k=5..66)"),
        Patch(facecolor=PBCH_COLOR, edgecolor=PBCH_COLOR, alpha=0.12, label="PBCH simboli (l=[7,8,9,10] po SF)"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.7,
    )
    leg.get_frame().set_linewidth(1.2)

    plt.tight_layout()

    out_dir = os.path.join("examples", "results", "rx")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rx_grid_magnitude_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Sačuvano: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
