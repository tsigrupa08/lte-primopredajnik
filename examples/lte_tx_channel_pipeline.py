from __future__ import annotations

import sys
import os

# ------------------------------------------------
# Dodaj root projekta u PYTHONPATH
# ------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from transmitter.LTETxChain import LTETxChain
from channel.lte_channel import LTEChannel
from transmitter.pbch import PBCHEncoder


# ================================================================
# Results folder: examples/results
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
# POMOĆNE FUNKCIJE
# ================================================================
def make_mib_bits(num_bits: int = 24) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=num_bits, dtype=np.int64)


def build_transmitter(
    n_id_2: int = 0,
    ndlrb: int = 6,
    num_subframes: int = 4,
    normal_cp: bool = True,
) -> LTETxChain:
    return LTETxChain(
        n_id_2=n_id_2,
        ndlrb=ndlrb,
        num_subframes=num_subframes,
        normal_cp=normal_cp,
    )


def generate_tx_waveform(
    tx: LTETxChain,
    mib_bits: np.ndarray,
) -> tuple[np.ndarray, float]:
    return tx.generate_waveform(mib_bits=mib_bits)


def build_channel(
    freq_offset_hz: float,
    fs: float,
    snr_db: float,
    seed: int | None = 123,
    initial_phase_rad: float = 0.0,
) -> LTEChannel:
    return LTEChannel(
        freq_offset_hz=freq_offset_hz,
        sample_rate_hz=fs,
        snr_db=snr_db,
        seed=seed,
        initial_phase_rad=initial_phase_rad,
    )


def apply_channel(ch: LTEChannel, x: np.ndarray) -> np.ndarray:
    ch.reset()
    return ch.apply(x)


def encode_pbch_symbols(mib_bits: np.ndarray) -> np.ndarray:
    """
    Enkodira PBCH iz MIB bitova i vraća QPSK simbole (TX konstelacija).
    """
    enc = PBCHEncoder()
    symbols = enc.encode(mib_bits)
    return symbols

# ================================================================
# PLOT FUNKCIJE (SA SNIMANJEM)
# ================================================================
def plot_waveforms(
    tx: np.ndarray,
    rx: np.ndarray,
    fs: float,
    num_samples: int = 2000,
) -> None:

    n = min(num_samples, tx.size, rx.size)
    t = np.arange(n) / fs

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, tx[:n].real, label="Tx Real")
    plt.plot(t, tx[:n].imag, label="Tx Imag", alpha=0.7)
    plt.title("LTE Tx waveform (segment)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(t, rx[:n].real, label="Rx Real")
    plt.plot(t, rx[:n].imag, label="Rx Imag", alpha=0.7)
    plt.title("LTE Rx waveform (after channel)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "lte_tx_rx_waveform_segment.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_constellation(
    symbols: np.ndarray,
    title: str,
    filename: str,
) -> None:

    plt.figure(figsize=(6, 6))
    plt.scatter(symbols.real, symbols.imag, s=12, alpha=0.7)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.axvline(0.0, color="gray", linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.axis("equal")

    plt.savefig(
        os.path.join(RESULTS_DIR, filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ================================================================
# MAIN
# ================================================================
def main() -> None:

    # 1) LTE predajni lanac
    tx_chain = build_transmitter(
        n_id_2=0,
        ndlrb=6,
        num_subframes=4,
        normal_cp=True,
    )

    # 2) MIB (24 bita)
    mib_bits = make_mib_bits(24)

    # 3) Tx OFDM waveform
    tx_waveform, fs = generate_tx_waveform(tx_chain, mib_bits)

    # 4) Kanal (offset + AWGN)
    ch = build_channel(
        freq_offset_hz=300.0,
        fs=fs,
        snr_db=10.0,
        seed=123,
    )

    # 5) Rx waveform
    rx_waveform = apply_channel(ch, tx_waveform)

    # 6) Snimi Tx/Rx waveform segment
    plot_waveforms(tx_waveform, rx_waveform, fs)

    # 7) PBCH konstelacija (TX)
    pbch_symbols = encode_pbch_symbols(mib_bits)
    plot_constellation(
        pbch_symbols,
        title="PBCH QPSK constellation (Tx)",
        filename="lte_pbch_constellation_tx.png",
    )

    print("LTE Tx → Channel example finished.")
    print("Results saved in examples/results/")


if __name__ == "__main__":
    main()
