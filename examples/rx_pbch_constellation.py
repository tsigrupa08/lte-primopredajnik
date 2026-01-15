from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt


def plot_pbch_constellation_rx(
    pbch_symbols: np.ndarray,
    out_path: Optional[Union[str, Path]] = None,
    *,
    show_ideal_qpsk: bool = True,
    title: str = "RX-only: PBCH konstelacija (izvučeni simboli prije demapiranja)",
    dpi: int = 180,
) -> str:
    """
    Prikazuje RX-only PBCH konstelaciju (izvučeni simboli prije demapiranja).

    Parameters
    ----------
    pbch_symbols : np.ndarray
        Kompleksni PBCH simboli izvučeni iz RX resource grida prije QPSK demapiranja.
        Tipično shape (960,) za normal CP i NDLRB=6 uz PBCH raspoređen preko 4 subfrejma.
    out_path : str | pathlib.Path | None, optional
        Putanja za spremanje slike. Ako je None, sprema u:
        `examples/results/rx/pbch_constellation_rx.png`.
        Ako je zadat samo naziv fajla (bez foldera), opet se sprema u `examples/results/rx/`.
    show_ideal_qpsk : bool, optional
        Ako je True, prikaži idealne QPSK tačke kao prazne krugove.
    title : str, optional
        Naslov figure.
    dpi : int, optional
        DPI za eksport.

    Returns
    -------
    str
        Apsolutna putanja do spremljene slike.
    """
    # -----------------------------
    # Putanja za spremanje (OBAVEZNO: examples/results/rx)
    # -----------------------------
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "examples" / "results" / "rx"
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        out_file = out_dir / "pbch_constellation_rx.png"
    else:
        out_path = Path(out_path)
        if out_path.parent == Path("."):
            out_file = out_dir / out_path.name
        else:
            out_file = out_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Ulazni simboli
    # -----------------------------
    x = np.asarray(pbch_symbols, dtype=np.complex128).ravel()
    x = x[x != (0.0 + 0.0j)]  # izbaci nule ako postoje

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6.6, 6.6))

    ax.scatter(
        x.real,
        x.imag,
        s=12,
        alpha=0.45,
        linewidths=0.0,
        label=f"PBCH RX simboli (N={x.size})",
    )

    # Idealne QPSK tačke (prazni krugovi)
    if show_ideal_qpsk:
        ideal = np.array(
            [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j],
            dtype=np.complex128,
        ) / np.sqrt(2.0)

        ax.scatter(
            ideal.real,
            ideal.imag,
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            label="Idealne QPSK tačke",
        )

    # -----------------------------
    # Stil
    # -----------------------------
    ax.set_title(title)
    ax.set_xlabel("I komponenta (realni dio)")
    ax.set_ylabel("Q komponenta (imaginarni dio)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")

    # Zum oko idealnih tačaka
    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([-1.25, 1.25])

    # Diskretna info-kutija (bez strelica)
    ax.text(
        0.02,
        0.98,
        "PBCH RX simboli izvučeni iz LTE resource grida\n(prije QPSK demapiranja)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.75),
    )

    ax.legend(loc="upper right")

    # -----------------------------
    # Spremanje
    # -----------------------------
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(out_file.resolve())


def _awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Dodaje AWGN na kompleksni signal za zadani SNR (dB), po snazi signala.

    Parameters
    ----------
    x : np.ndarray
        Ulazni kompleksni signal.
    snr_db : float
        Signal-to-noise ratio u dB.
    rng : np.random.Generator
        Generator za šum.

    Returns
    -------
    np.ndarray
        Signal sa dodanim AWGN šumom.
    """
    x = np.asarray(x)
    p_sig = np.mean(np.abs(x) ** 2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_noise = p_sig / snr_lin
    n = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) * np.sqrt(p_noise / 2.0)
    return x + n


if __name__ == "__main__":
    # ==========================================================
    # LTE SYSTEM-BASED (NE GENERIČKI):
    #   LTETxChain -> AWGN -> OFDM demod -> PBCHExtractor -> plot
    # ==========================================================
    #
    # Pokreni iz ROOT foldera projekta:
    #   python -m examples.rx_pbch_constellation
    #

    from transmitter.LTETxChain import LTETxChain
    from receiver.OFDM_demodulator import OFDMDemodulator
    from receiver.resource_grid_extractor import (
        PBCHConfig,
        PBCHExtractor,
        pbch_symbol_indices_for_subframes,
    )

    rng = np.random.default_rng(0)

    # LTE parametri (kao u projektu)
    ndlrb = 6
    normal_cp = True
    pci = 0

    # TX mora imati >= 4 subfrejma da mapira 960 PBCH simbola (4×240)
    tx = LTETxChain(n_id_2=pci, ndlrb=ndlrb, num_subframes=4, normal_cp=normal_cp)
    mib_bits = rng.integers(0, 2, size=24, dtype=np.uint8)
    tx_waveform, fs = tx.generate_waveform(mib_bits=mib_bits)

    # Channel: AWGN (da se vidi oblak)
    snr_db = 20.0
    rx_waveform = _awgn(tx_waveform, snr_db=snr_db, rng=rng)

    # RX: OFDM demod + aktivni subcarriers (72)
    ofdm = OFDMDemodulator(ndlrb=ndlrb, normal_cp=normal_cp)
    grid_full = ofdm.demodulate(rx_waveform)
    grid_active = ofdm.extract_active_subcarriers(grid_full)

    # PBCH extraction: 240 po subfrejmu × 4 subfrejma = 960 simbola
    pbch_syms_parts = []
    for sf in range(4):
        idx_sf = pbch_symbol_indices_for_subframes(
            num_subframes=1,
            normal_cp=normal_cp,
            start_subframe=sf,
        )
        cfg_sf = PBCHConfig(
            ndlrb=ndlrb,
            normal_cp=normal_cp,
            pbch_symbol_indices=idx_sf,
            pbch_symbols_to_extract=240,
        )
        pbch_syms_parts.append(PBCHExtractor(cfg_sf).extract(grid_active))

    pbch_syms = np.concatenate(pbch_syms_parts)  # (960,)

    # Plot + save (OBAVEZNO u examples/results/rx)
    out = plot_pbch_constellation_rx(
        pbch_syms,
        out_path="pbch_constellation_rx.png",
        show_ideal_qpsk=True,
    )
    print("Spremljeno:", out)
