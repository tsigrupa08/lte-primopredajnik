# examples/rx_pss_correlation_demo.py
"""
RX-only: PSS korelacija (3 krive) + odabir N_ID_2

Šta treba da vidiš na slici (očekivano):
- Imaš 3 subplota: N_ID_2 = 0, 1, 2 (tri kandidata PSS sekvence).
- Na JEDNOM subplotu treba biti dominantan peak oko stvarne pozicije PSS-a:
    - To je onaj subplot čiji je N_ID_2 jednak N_ID_2_hat (označeno "(odabrano)").
- Vertikalna isprekidana crna linija = detektovani tau_hat.
- Siva tačkasta linija (ako je prikazana) = očekivani položaj (tau_expected) za ovaj demo.
- Y osa je rel dB: maksimum svake krive je 0 dB → peak se odmah vidi, šum je negativan.

Kako pokrenuti:
    python examples/rx_pss_correlation_demo.py

Output:
    examples/results/rx/pss_correlation_metrics_3subplots_relDB.png
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Path setup (da radi i kad se pokreće direktno iz examples/)
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Imports iz projekta
# ---------------------------------------------------------------------
from transmitter.LTETxChain import LTETxChain
from transmitter.ofdm import OFDMModulator



from channel.lte_channel import LTEChannel



def symbol_start_indices(ofdm: OFDMModulator) -> list[int]:
    """Start indeksi OFDM simbola u waveform-u."""
    starts: list[int] = []
    idx = 0
    for sym_idx in range(ofdm.num_ofdm_symbols):
        starts.append(idx)
        cp_len = int(ofdm.cp_lengths[sym_idx % ofdm.n_symbols_per_slot])
        idx += ofdm.N + cp_len
    return starts


def build_pss_template_time(n_id_2: int, ndlrb: int, normal_cp: bool) -> tuple[np.ndarray, float, int]:
    """
    Napravi TX waveform samo sa PSS (1 subframe), pa izvadi vremenski segment PSS simbola (CP+FFT dio).
    Vraća: (template, fs, pss_symbol_start_index_u_subframe_waveformu).
    """
    tx = LTETxChain(n_id_2=n_id_2, ndlrb=ndlrb, num_subframes=1, normal_cp=normal_cp)
    tx_waveform, fs = tx.generate_waveform(mib_bits=None)

    ofdm = OFDMModulator(tx.grid)
    starts = symbol_start_indices(ofdm)

    # Normal CP: PSS je u simbolu l=6 (prvi slot). Extended CP: l=5
    pss_sym = 6 if normal_cp else 5
    pss_start = starts[pss_sym]

    cp_len = int(ofdm.cp_lengths[pss_sym % ofdm.n_symbols_per_slot])
    pss_len = ofdm.N + cp_len

    template = tx_waveform[pss_start:pss_start + pss_len].astype(np.complex64)
    return template, float(fs), int(pss_start)


def correlate_templates_norm(rx: np.ndarray, templates: dict[int, np.ndarray]) -> tuple[np.ndarray, int, int]:
    """
    Normalizovana korelacija:
        metric(tau) = |sum r[tau+n] * conj(t[n])| / (sqrt(sum|r|^2) * sqrt(sum|t|^2))

    Bitno:
    - np.correlate za kompleksne nizove već radi conj nad drugim argumentom,
      zato se poziva kao np.correlate(rx, t), a NE np.conj(t).
    """
    nids = sorted(templates.keys())
    L = templates[nids[0]].size
    corr_len = rx.size - L + 1
    if corr_len <= 0:
        raise ValueError("RX signal je prekratak za korelaciju sa PSS template-om.")

    # Sliding energija RX prozora: sum |rx[tau:tau+L]|^2
    rx_energy = np.convolve(np.abs(rx) ** 2, np.ones(L, dtype=np.float64), mode="valid").astype(np.float64)
    rx_energy = np.maximum(rx_energy, 1e-12)

    metrics = np.zeros((len(nids), corr_len), dtype=np.float64)

    for i, nid in enumerate(nids):
        t = templates[nid].astype(np.complex64)
        t_energy = float(np.sum(np.abs(t) ** 2))
        t_energy = max(t_energy, 1e-12)

        c = np.correlate(rx, t, mode="valid")  # ispravno za kompleksne
        metrics[i, :] = np.abs(c) / np.sqrt(rx_energy * t_energy)

    flat_idx = int(np.argmax(metrics))
    i_hat, tau_hat = np.unravel_index(flat_idx, metrics.shape)
    nid_hat = nids[i_hat]
    return metrics, int(tau_hat), int(nid_hat)


def plot_three_subplots(metrics: np.ndarray,
                        tau_hat: int,
                        nid_hat: int,
                        fs: float,
                        out_path: Path,
                        tau_expected: int | None,
                        half_win: int = 1200,
                        smooth_len: int = 21) -> None:
    """
    Pravi jedan graf sa 3 subplota (rel dB), zumiran oko tau_hat.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # X osa u ms
    x = np.arange(metrics.shape[1])
    t_ms = x / fs * 1e3

    # Zum oko detektovanog peaka
    x0 = max(0, tau_hat - half_win)
    x1 = min(metrics.shape[1] - 1, tau_hat + half_win)

    # Smoothing samo radi ljepšeg prikaza (ne utiče na odluku)
    if smooth_len > 1:
        w = np.ones(smooth_len, dtype=np.float64) / smooth_len
        metrics_vis = np.vstack([np.convolve(metrics[i], w, mode="same") for i in range(3)])
    else:
        metrics_vis = metrics

    eps = 1e-12
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    nids = (0, 1, 2)

    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    for i, nid in enumerate(nids):
        m = metrics_vis[i]
        # Relativni dB: peak svake krive je ~0 dB
        m_rel_db = 20.0 * np.log10((m + eps) / (np.max(m) + eps))

        ax = axs[i]
        ax.plot(t_ms[x0:x1 + 1], m_rel_db[x0:x1 + 1], color=colors[nid], linewidth=2.0, label=f"N_ID_2={nid}")

        # Detektovani peak (tau_hat) – isti na sva 3 subplota radi poređenja
        ax.axvline(tau_hat / fs * 1e3, color="k", linestyle="--", linewidth=1.6, label="tau_hat")
        ax.scatter([tau_hat / fs * 1e3], [m_rel_db[tau_hat]], color=colors[nid], s=45, zorder=5)

        # Očekivana pozicija (za ovaj demo)
        if tau_expected is not None:
            ax.axvline(tau_expected / fs * 1e3, color="gray", linestyle=":", linewidth=1.2, label="tau_expected")

        ax.set_title(f"N_ID_2={nid}" + (" (odabrano)" if nid == nid_hat else ""))
        ax.set_ylabel("rel dB")
        ax.set_ylim(-35, 2)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

        # Kratka “legenda” na samom plotu (šta je šta)
        if i == 0:
            ax.text(
                0.01, 0.05,
                "Očekivano: samo jedan subplot ima dominantan peak ~0 dB.\n"
                "Pobjednik = N_ID_2_hat (označen kao 'odabrano').",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom"
            )

    fig.suptitle(f"PSS korelacija (3 subplota, rel dB) | N_ID_2_hat={nid_hat}, tau_hat={tau_hat}", y=0.98)
    axs[-1].set_xlabel("Vrijeme (ms)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()

    print(f"[OK] Sačuvano: {out_path}")


def main():
    # -----------------------------------------------------------------
    # Parametri DEMO-a (igraj se s ovim da vidiš ponašanje)
    # -----------------------------------------------------------------
    ndlrb = 6
    normal_cp = True

    true_nid2 = 1
    timing_offset_samples = 4000
    cfo_hz = 200.0
    snr_db = 5.0
    seed = 123

    # Za “čist” demo (jedan očekivani peak), drži num_subframes=1
    num_subframes = 1

    # -----------------------------------------------------------------
    # TX waveform (samo PSS)
    # -----------------------------------------------------------------
    tx = LTETxChain(n_id_2=true_nid2, ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=normal_cp)
    tx_waveform, fs = tx.generate_waveform(mib_bits=None)
    tx_waveform = tx_waveform.astype(np.complex64)

    # -----------------------------------------------------------------
    # RX = timing offset + kanal (CFO + AWGN)
    # -----------------------------------------------------------------
    rx = np.concatenate([np.zeros(timing_offset_samples, dtype=np.complex64), tx_waveform])

    ch = LTEChannel(
        freq_offset_hz=cfo_hz,
        sample_rate_hz=fs,
        snr_db=snr_db,
        seed=seed,
        initial_phase_rad=0.0,
    )
    ch.reset()
    rx = ch.apply(rx).astype(np.complex64)

    # -----------------------------------------------------------------
    # Napravi 3 template-a (time-domain segment PSS simbola)
    # -----------------------------------------------------------------
    templates: dict[int, np.ndarray] = {}
    pss_symbol_start = None

    for nid in (0, 1, 2):
        t, fs_t, pss_start = build_pss_template_time(nid, ndlrb=ndlrb, normal_cp=normal_cp)
        if abs(fs_t - fs) > 1e-9:
            raise RuntimeError("FS mismatch između template-a i RX.")
        templates[nid] = t
        if nid == true_nid2:
            pss_symbol_start = pss_start

    # -----------------------------------------------------------------
    # Korelacija + odluka
    # -----------------------------------------------------------------
    metrics, tau_hat, nid_hat = correlate_templates_norm(rx, templates)

    tau_expected = None
    if pss_symbol_start is not None:
        tau_expected = timing_offset_samples + pss_symbol_start

    print("---- PSS SYNC DEMO ----")
    print(f"fs = {fs:.0f} Hz")
    print(f"true N_ID_2 = {true_nid2}")
    print(f"snr_db = {snr_db}, cfo_hz = {cfo_hz}, timing_offset_samples = {timing_offset_samples}")
    print(f"N_ID_2_hat = {nid_hat}")
    print(f"tau_hat = {tau_hat} samples")
    if tau_expected is not None:
        print(f"tau_expected (ideal) ≈ {tau_expected} samples")

    # -----------------------------------------------------------------
    # Plot + save (samo 3 subplota)
    # -----------------------------------------------------------------
    out_dir = PROJECT_ROOT / "examples" / "results" / "rx"
    out_path = out_dir / "pss_correlation_metrics_3subplots_relDB.png"

    plot_three_subplots(
        metrics=metrics,
        tau_hat=tau_hat,
        nid_hat=nid_hat,
        fs=fs,
        out_path=out_path,
        tau_expected=tau_expected,
        half_win=1200,   # zoom
        smooth_len=21,   # smoothing za ljepši prikaz
    )


if __name__ == "__main__":
    main()
