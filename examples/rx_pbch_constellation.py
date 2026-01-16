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

    x = np.asarray(pbch_symbols, dtype=np.complex128).ravel()
    x = x[x != (0.0 + 0.0j)]

    fig, ax = plt.subplots(figsize=(6.6, 6.6))

    ax.scatter(
        x.real,
        x.imag,
        s=12,
        alpha=0.45,
        linewidths=0.0,
        label=f"PBCH RX simboli (N={x.size})",
    )

    if show_ideal_qpsk:
        ideal = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j], dtype=np.complex128) / np.sqrt(2.0)
        ax.scatter(
            ideal.real,
            ideal.imag,
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            label="Idealne QPSK tačke",
        )

    ax.set_title(title)
    ax.set_xlabel("I komponenta (realni dio)")
    ax.set_ylabel("Q komponenta (imaginarni dio)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([-1.25, 1.25])

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

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(out_file.resolve())


def _extract_pbch_symbols_from_rx_waveform(
    rx_waveform: np.ndarray,
    *,
    ndlrb: int,
    normal_cp: bool,
) -> np.ndarray:
    """
    OFDM demod + PBCHExtractor: vrati PBCH simbole (prije demapiranja).

    Parameters
    ----------
    rx_waveform : np.ndarray
        Primljeni kompleksni OFDM waveform (nakon kanala).
    ndlrb : int
        Broj downlink RB.
    normal_cp : bool
        Normal CP ili extended.

    Returns
    -------
    np.ndarray
        Kompleksni PBCH simboli, shape (960,).
    """
    from receiver.OFDM_demodulator import OFDMDemodulator
    from receiver.resource_grid_extractor import PBCHConfig, PBCHExtractor, pbch_symbol_indices_for_subframes

    ofdm = OFDMDemodulator(ndlrb=ndlrb, normal_cp=normal_cp)
    grid_full = ofdm.demodulate(rx_waveform)
    grid_active = ofdm.extract_active_subcarriers(grid_full)

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

    return np.concatenate(pbch_syms_parts)  # (960,)


if __name__ == "__main__":
    # ==========================================================
    # LTE SYSTEM-BASED + VAŠ KANAL:
    #   LTETxChain -> LTEChannel(freq_offset + awgn) -> RX extract -> 2 plota
    # ==========================================================
    #
    # Pokreni iz ROOT foldera:
    #   python -m examples.rx_pbch_constellation
    #

    from transmitter.LTETxChain import LTETxChain
    from channel.lte_channel import LTEChannel

    rng = np.random.default_rng(0)

    # LTE parametri
    ndlrb = 6
    normal_cp = True
    pci = 0

    # TX mora imati 4 subfrejma (960 PBCH simbola)
    tx = LTETxChain(n_id_2=pci, ndlrb=ndlrb, num_subframes=4, normal_cp=normal_cp)
    mib_bits = rng.integers(0, 2, size=24, dtype=np.uint8)
    tx_waveform, fs = tx.generate_waveform(mib_bits=mib_bits)

    # --------------------------
    # Scenario 1: dobar kanal
    # --------------------------
    ch1 = LTEChannel(
        freq_offset_hz=0.0,
        sample_rate_hz=fs,
        snr_db=20.0,
        seed=1,
    )
    rx1 = ch1.apply(tx_waveform)
    pbch1 = _extract_pbch_symbols_from_rx_waveform(rx1, ndlrb=ndlrb, normal_cp=normal_cp)

    out1 = plot_pbch_constellation_rx(
        pbch1,
        out_path="pbch_constellation_rx_snr20_cfo0.png",
        show_ideal_qpsk=True,
        title="RX-only: PBCH konstelacija (SNR=20 dB, CFO=0 Hz)",
    )
    print("Spremljeno:", out1)

    # --------------------------
    # Scenario 2: lošiji kanal
    # --------------------------
    # CFO izabran da se vidi rotacija/rasipanje 
    ch2 = LTEChannel(
        freq_offset_hz=500.0,
        sample_rate_hz=fs,
        snr_db=8.0,
        seed=2,
    )
    rx2 = ch2.apply(tx_waveform)
    pbch2 = _extract_pbch_symbols_from_rx_waveform(rx2, ndlrb=ndlrb, normal_cp=normal_cp)

    out2 = plot_pbch_constellation_rx(
        pbch2,
        out_path="pbch_constellation_rx_snr8_cfo500.png",
        show_ideal_qpsk=True,
        title="RX-only: PBCH konstelacija (SNR=8 dB, CFO=500 Hz)",
    )
    print("Spremljeno:", out2)
