# examples/rx_cfo_correction_demo.py
"""
C2) RX-only: CFO procjena i korekcija (LTE)

Šta treba da vidiš:
1) GORNJI graf (prije korekcije):
   - Crna isprekidana linija je DC (0 Hz).
   - Crvena tačkasta linija je CFO_true (što je dato u kanalu).
   - Narandžasta isprekidano-tačkasta linija je cfo_hat (procjena).
   Očekivanje: cfo_hat treba biti blizu CFO_true.

2) DONJI graf (poslije korekcije):
   - Crna isprekidana linija je DC (0 Hz).
   - Ljubičasta tačkasta linija je residual CFO = CFO_true - cfo_hat.
   Očekivanje: residual treba biti blizu 0 Hz (tj. linija skoro na DC).

Kako pokrenuti:
    python examples/rx_cfo_correction_demo.py

Output:
    examples/results/rx/cfo_fft_before_after.png

Napomena:
- Da bi se CFO vizualno vidio na FFT, moraš ZUMIRATI oko 0 Hz (npr. ±100 kHz),
  jer je CFO 5 kHz sitan u odnosu na ±800 kHz.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transmitter.LTETxChain import LTETxChain
from transmitter.ofdm import OFDMModulator
from receiver.pss_sync import PSSSynchronizer
from channel.lte_channel import LTEChannel


def symbol_start_indices(ofdm: OFDMModulator) -> list[int]:
    starts = []
    idx = 0
    for sym_idx in range(ofdm.num_ofdm_symbols):
        starts.append(idx)
        cp_len = int(ofdm.cp_lengths[sym_idx % ofdm.n_symbols_per_slot])
        idx += ofdm.N + cp_len
    return starts


def build_pss_symbol_info(ndlrb: int, normal_cp: bool, n_id_2_for_tx: int) -> dict:
    tx = LTETxChain(n_id_2=n_id_2_for_tx, ndlrb=ndlrb, num_subframes=1, normal_cp=normal_cp)
    tx_waveform, fs = tx.generate_waveform(mib_bits=None)
    tx_waveform = tx_waveform.astype(np.complex64)

    ofdm = OFDMModulator(tx.grid)
    starts = symbol_start_indices(ofdm)

    pss_sym = 6 if normal_cp else 5
    pss_start = int(starts[pss_sym])
    cp_len = int(ofdm.cp_lengths[pss_sym % ofdm.n_symbols_per_slot])
    N = int(ofdm.N)
    pss_len = cp_len + N

    return {
        "tx_waveform": tx_waveform,
        "fs": float(fs),
        "ofdm_N": N,
        "cp_len_pss": cp_len,
        "pss_start": pss_start,
        "pss_len": pss_len,
    }


def fft_mag_db(x: np.ndarray, fs: float, nfft: int, window: bool = True) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.complex64)

    if window:
        w = np.hanning(x.size).astype(np.float32)
        xw = x * w
    else:
        xw = x

    X = np.fft.fftshift(np.fft.fft(xw, n=nfft))
    mag = np.abs(X).astype(np.float64)

    eps = 1e-12
    mag_db = 20.0 * np.log10(mag + eps)
    mag_db -= np.max(mag_db)  # rel: max=0 dB

    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return f, mag_db


def main():
    # --------------------------------------------------
    # DEMO parametri
    # --------------------------------------------------
    ndlrb = 6
    normal_cp = True
    true_nid2 = 1

    CFO_true = 5000.0
    snr_db = 25.0
    timing_offset = 3000

    # FFT parametri
    seg_len = 4096
    nfft = 16384

    # ZOOM: automatski da CFO bude “vidljiv”
    # (za CFO=5kHz -> zoom ~100kHz)
    f_zoom_hz = max(50e3, 20.0 * abs(CFO_true))

    # --------------------------------------------------
    # 1) TX
    # --------------------------------------------------
    info = build_pss_symbol_info(ndlrb=ndlrb, normal_cp=normal_cp, n_id_2_for_tx=true_nid2)
    tx_waveform = info["tx_waveform"]
    fs = info["fs"]

    # --------------------------------------------------
    # 2) RX: timing offset + kanal (CFO_true + AWGN)
    # --------------------------------------------------
    tx_delayed = np.concatenate([np.zeros(timing_offset, dtype=np.complex64), tx_waveform])

    ch = LTEChannel(
        freq_offset_hz=CFO_true,
        sample_rate_hz=fs,
        snr_db=snr_db,
        seed=123,
    )
    rx = ch.apply(tx_delayed).astype(np.complex64)

    # --------------------------------------------------
    # 3) PSS sync: tau_hat, nid_hat, cfo_hat
    # --------------------------------------------------
    sync = PSSSynchronizer(sample_rate_hz=fs)
    corr = sync.correlate(rx)
    tau_hat, nid_hat = sync.estimate_timing(corr)
    cfo_hat = sync.estimate_cfo(rx, tau_hat, nid_hat)

    rx_corr = sync.apply_cfo_correction(rx, cfo_hat)

    # --------------------------------------------------
    # 4) Segment za FFT (oko tau_hat)
    # --------------------------------------------------
    start = int(max(0, tau_hat))
    end = int(min(rx.size, start + seg_len))
    seg = rx[start:end]
    seg2 = rx_corr[start:end]

    if seg.size < seg_len:
        pad = seg_len - seg.size
        seg = np.concatenate([seg, np.zeros(pad, dtype=np.complex64)])
        seg2 = np.concatenate([seg2, np.zeros(pad, dtype=np.complex64)])

    f, S_db = fft_mag_db(seg, fs=fs, nfft=nfft, window=True)
    _, S2_db = fft_mag_db(seg2, fs=fs, nfft=nfft, window=True)

    # --------------------------------------------------
    # 5) Ispis
    # --------------------------------------------------
    err = float(cfo_hat - CFO_true)
    residual = float(CFO_true - cfo_hat)

    print("---- CFO DEMO ----")
    print(f"fs = {fs:.0f} Hz | ndlrb={ndlrb} | normal_cp={normal_cp}")
    print(f"N_ID_2_hat = {nid_hat} | tau_hat = {tau_hat} samples")
    print(f"CFO_true = {CFO_true:.2f} Hz")
    print(f"cfo_hat  = {cfo_hat:.2f} Hz")
    print(f"error    = {err:.2f} Hz")
    print(f"residual(after) = {residual:.2f} Hz")

    # --------------------------------------------------
    # 6) Plot (2 subplota: prije/poslije) + jasni markeri
    # --------------------------------------------------
    out_dir = PROJECT_ROOT / "examples" / "results" / "rx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cfo_fft_before_after.png"

    # Prikaži x-osu u kHz (čitljivije)
    fk = f / 1e3
    zoom_khz = f_zoom_hz / 1e3

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # --- prije ---
    axs[0].plot(fk, S_db, linewidth=1.8)
    axs[0].axvline(0.0, color="k", linestyle="--", linewidth=1.2, label="DC (0 Hz)")
    axs[0].axvline(CFO_true / 1e3, color="tab:red", linestyle=":", linewidth=1.8, label="CFO_true")
    axs[0].axvline(cfo_hat / 1e3, color="tab:orange", linestyle="-.", linewidth=1.8, label="cfo_hat")
    axs[0].set_title("Prije CFO korekcije: |FFT| (dB, rel.)")
    axs[0].set_ylabel("|FFT| [dB]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")

    axs[0].text(
        0.01, 0.05,
        "Očekivanje: cfo_hat ≈ CFO_true.\n"
        "CFO znači da se signal u frekvenciji ponaša kao 'shift' za CFO.",
        transform=axs[0].transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    # --- poslije ---
    axs[1].plot(fk, S2_db, linewidth=1.8)
    axs[1].axvline(0.0, color="k", linestyle="--", linewidth=1.2, label="DC (0 Hz)")
    axs[1].axvline(residual / 1e3, color="tab:purple", linestyle=":", linewidth=1.8, label="residual = CFO_true - cfo_hat")
    axs[1].set_title("Poslije CFO korekcije: |FFT| (dB, rel.)")
    axs[1].set_xlabel("Frekvencija [kHz]")
    axs[1].set_ylabel("|FFT| [dB]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    axs[1].text(
        0.01, 0.05,
        "Očekivanje: residual ≈ 0 Hz (linija skoro na DC).\n"
        "To znači da je CFO uspješno kompenzovan.",
        transform=axs[1].transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    # Zoom oko DC da se CFO jasno vidi
    axs[1].set_xlim(-zoom_khz, zoom_khz)

    fig.suptitle(
        f"CFO: true={CFO_true:.1f} Hz | hat={cfo_hat:.1f} Hz | err={err:.1f} Hz | "
        f"residual={residual:.1f} Hz | N_ID_2_hat={nid_hat} | tau_hat={tau_hat}",
        y=0.98
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"[OK] Slika sačuvana: {out_path}")


if __name__ == "__main__":
    main()
