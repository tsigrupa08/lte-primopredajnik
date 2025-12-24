"""
tx_chain_demo
=============

Primjer kompletnog LTE predajnog lanca (TX chain) i vizualizacije:

- PBCH QPSK konstelacija
- OFDM talasni oblik (real, imag, |s[n]|)

TX-only demonstracija (bez kanala).
"""

from __future__ import annotations

import os
import sys
import inspect
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Dodaj root projekta u PYTHONPATH
# ------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transmitter.LTETxChain import LTETxChain
from transmitter.pbch import PBCHEncoder

# ================================================================
# Results folder (TX-only)
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = os.path.join(BASE_DIR, "results")
TX_DIR = os.path.join(RESULTS_BASE, "tx")
os.makedirs(TX_DIR, exist_ok=True)

# ================================================================
# PBCH helper
# ================================================================
def _make_pbch_encoder(target_bits: int = 1920) -> PBCHEncoder:
    sig = inspect.signature(PBCHEncoder)
    params = sig.parameters
    kwargs = {}

    if "target_bits" in params:
        kwargs["target_bits"] = target_bits
    if "verbose" in params:
        kwargs["verbose"] = False

    return PBCHEncoder(**kwargs)


def _pbch_symbols_from_mib(mib_bits: np.ndarray) -> np.ndarray:
    encoder = _make_pbch_encoder(target_bits=1920)
    syms = encoder.encode(mib_bits)
    return np.asarray(syms, dtype=np.complex128).flatten()

# ================================================================
# PLOT FUNKCIJE
# ================================================================
def plot_pbch_qpsk_constellation(pbch_symbols: np.ndarray) -> None:
    syms = np.asarray(pbch_symbols).flatten()
    syms = syms[np.isfinite(syms)]
    syms = syms[np.abs(syms) > 1e-12]

    if syms.size == 0:
        raise ValueError("Nema PBCH simbola za plot.")

    syms = syms / np.sqrt(np.mean(np.abs(syms) ** 2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(syms.real, syms.imag, s=16, alpha=0.6)

    ideal = np.array([1+1j, 1-1j, -1-1j, -1+1j]) / np.sqrt(2)
    for pt in ideal:
        ax.scatter(pt.real, pt.imag, marker="x", s=100, linewidths=2)

    ax.set_title("PBCH QPSK konstelacija (TX)")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    fig.savefig(
        os.path.join(TX_DIR, "tx_chain_pbch_constellation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ofdm_time_segment(
    waveform: np.ndarray,
    sample_rate: float,
    num_samples: int = 4000,
) -> None:
    """
    OFDM talasni oblik – tri odvojena prikaza:
    1) realni dio
    2) imaginarni dio
    3) apsolutna vrijednost |s[n]|
    """
    seg = waveform[:num_samples]
    t = np.arange(seg.size) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # --- Realni dio ---
    axes[0].plot(t, seg.real, color="tab:blue")
    axes[0].set_title("OFDM – realni dio (TX)")
    axes[0].set_ylabel("Amplituda")
    axes[0].grid(True, alpha=0.4)

    # --- Imaginarni dio ---
    axes[1].plot(t, seg.imag, color="tab:orange")
    axes[1].set_title("OFDM – imaginarni dio (TX)")
    axes[1].set_ylabel("Amplituda")
    axes[1].grid(True, alpha=0.4)

    # --- Apsolutna vrijednost ---
    axes[2].plot(t, np.abs(seg), color="tab:green")
    axes[2].set_title("OFDM – |s[n]| (TX)")
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("Amplituda")
    axes[2].grid(True, alpha=0.4)

    fig.tight_layout()

    fig.savefig(
        os.path.join(TX_DIR, "tx_chain_ofdm_time_segment.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# ================================================================
# MAIN
# ================================================================
def run_tx_chain_demo(
    n_id_2: int = 0,
    ndlrb: int = 6,
    num_subframes: int = 4,
    normal_cp: bool = True,
    mib_bits: Optional[Iterable[int]] = None,
) -> None:

    if mib_bits is None:
        mib = np.random.randint(0, 2, 24, dtype=np.uint8)
    else:
        mib = np.array(list(mib_bits), dtype=np.uint8)
        if mib.size != 24:
            raise ValueError("MIB mora imati tačno 24 bita.")

    tx = LTETxChain(
        n_id_2=n_id_2,
        ndlrb=ndlrb,
        num_subframes=num_subframes,
        normal_cp=normal_cp,
    )

    waveform, fs = tx.generate_waveform(mib_bits=mib)
    pbch_symbols = _pbch_symbols_from_mib(mib)

    print(f"[TX] MIB bits: {mib}")
    print(f"[TX] Waveform length: {waveform.size}, fs = {fs} Hz")

    plot_pbch_qpsk_constellation(pbch_symbols)
    plot_ofdm_time_segment(waveform, fs)

    print("[OK] TX chain demo završen.")
    print("Rezultati su u examples/results/tx/")


if __name__ == "__main__":
    run_tx_chain_demo()
