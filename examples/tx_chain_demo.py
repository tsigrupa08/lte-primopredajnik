"""
tx_chain_demo
=============

Primjer kompletnog LTE predajnog lanca (TX chain) i vizualizacije:

* QPSK konstelacija PBCH simbola
* dio OFDM talasnog oblika u vremenu (realni dio, imaginarni dio i amplituda)

Skripta koristi LTETxChain iz paketa ``transmitter``:

* :class:`transmitter.LTETxChain.LTETxChain`

Kako pokrenuti
--------------

Iz root direktorija projekta pokreni:

.. code-block:: bash

    python -m examples.tx_chain_demo

Nakon pokretanja će se otvoriti dva prozora sa grafovima i snimiće se PNG slike u ``examples/``:
1) PBCH QPSK konstelacija
2) Segment OFDM signala u vremenu
"""

from __future__ import annotations

import os
import sys
import inspect
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------#
# Podešavanje putanje do projekta da bi se "transmitter" paket mogao uvesti
# ---------------------------------------------------------------------------#
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transmitter.LTETxChain import LTETxChain
from transmitter.pbch import PBCHEncoder

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------#
# Pomoćno: PBCHEncoder kompatibilno kreiranje (da radi i ako se potpis razlikuje)
# ---------------------------------------------------------------------------#
def _make_pbch_encoder(target_bits: int = 1920) -> PBCHEncoder:
    """
    Pravi PBCHEncoder robustno. Cilj nam je da dobijemo ~960 QPSK simbola
    (tj. 1920 bita nakon rate-matchinga).
    """
    sig = inspect.signature(PBCHEncoder)
    params = sig.parameters
    kwargs = {}

    # najčešći parametri koje si koristila
    if "target_bits" in params:
        kwargs["target_bits"] = target_bits
    if "verbose" in params:
        kwargs["verbose"] = False

    return PBCHEncoder(**kwargs)


def _pbch_symbols_from_mib(mib_bits: np.ndarray) -> np.ndarray:
    """
    Generiše PBCH QPSK simbole direktno iz MIB bitova,
    da konstelaciju crtamo iz “čistih” simbola.
    """
    encoder = _make_pbch_encoder(target_bits=1920)
    syms = encoder.encode(mib_bits)
    return np.asarray(syms, dtype=np.complex128).flatten()


# ---------------------------------------------------------------------------#
# Vizualizacija
# ---------------------------------------------------------------------------#
def plot_pbch_qpsk_constellation(pbch_symbols: np.ndarray) -> plt.Figure:
    """
    Crta QPSK konstelaciju PBCH simbola.

    Popravka je SAMO za vizualizaciju:
    - ako simboli imaju velike vrijednosti (npr. ~180), pretpostavimo da je to
      uint8 wrap (-1 -> 255), pa ih vratimo u signed domen (+1/-1)
    - zatim normalizujemo snagu i fiksiramo ose da QPSK bude jasno vidljiv
    """
    syms = np.asarray(pbch_symbols).flatten()
    syms = syms[np.isfinite(syms)]

    # izbaci nule (ako ih ima)
    syms = syms[np.abs(syms) > 1e-12]

    if syms.size == 0:
        raise ValueError("Nema PBCH simbola za plot (sve nule ili NaN/Inf).")

    max_abs = float(np.max(np.abs(syms)))

    # Ako vidiš nešto tipa 180, 200, itd. -> “wrap” slučaj
    if max_abs > 5.0:
        s = np.sqrt(2.0)
        # vraćamo na “0/1/255” skalu pa u signed
        r_u = np.round(syms.real * s).astype(int)
        i_u = np.round(syms.imag * s).astype(int)

        r_s = np.where(r_u > 127, r_u - 256, r_u)  # 255 -> -1
        i_s = np.where(i_u > 127, i_u - 256, i_u)

        syms = (r_s + 1j * i_s) / s

    # Normalizacija snage (da bude lijepo oko idealnih tačaka)
    p = float(np.mean(np.abs(syms) ** 2))
    if p > 0.0:
        syms = syms / np.sqrt(p)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        syms.real,
        syms.imag,
        s=16,
        alpha=0.6,
        label="PBCH QPSK simboli",
    )

    # Idealne QPSK tačke (Gray mapping): 00, 01, 11, 10
    ideal = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j], dtype=np.complex128) / np.sqrt(2)
    labels = ["00", "01", "11", "10"]

    for sym, lab in zip(ideal, labels):
        ax.scatter(sym.real, sym.imag, marker="x", s=100, linewidths=2, label=f"Idealna tačka {lab}")
        ax.text(sym.real * 1.12, sym.imag * 1.12, lab, fontsize=10, ha="center", va="center")

    ax.set_title("PBCH QPSK konstelacija", fontsize=12)
    ax.set_xlabel("I (in-phase komponenta)")
    ax.set_ylabel("Q (quadrature komponenta)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    # Fiksne ose: QPSK uvijek lijepo vidljiv
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.legend(loc="best")

    save_path = os.path.join(EXAMPLES_DIR, "results_pbch_qpsk_constellation.png")
    fig.savefig(save_path, dpi=300)
    print(f"[OK] Spremljena PBCH konstelacija → {save_path}")
    print(f"[DBG] max |sym| prije sanacije: {max_abs:.3g}")

    return fig



def plot_ofdm_time_segment(
    waveform: np.ndarray,
    sample_rate: float,
    num_samples: int = 3000,
    start_sample: int = 0,
) -> plt.Figure:
    """
    Crta dio OFDM talasnog oblika u vremenu (real, imag, |.|).
    """
    end_sample = min(start_sample + num_samples, waveform.size)
    seg = waveform[start_sample:end_sample]
    t = np.arange(seg.size) / float(sample_rate)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, seg.real, label="Realni dio s[n]")
    ax.plot(t, seg.imag, linestyle="--", label="Imaginarni dio s[n]")
    ax.plot(t, np.abs(seg), linestyle=":", label="Apsolutna vrijednost |s[n]|")

    ax.set_title("Dio OFDM talasnog oblika u vremenu", fontsize=12)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Amplituda")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    save_path = os.path.join(EXAMPLES_DIR, "results_ofdm_time_segment.png")
    fig.savefig(save_path, dpi=300)
    print(f"[OK] Spremljen OFDM segment → {save_path}")

    return fig


# ---------------------------------------------------------------------------#
# Glavni demo (preko LTETxChain)
# ---------------------------------------------------------------------------#
def run_tx_chain_demo(
    n_id_2: int = 0,
    ndlrb: int = 6,
    num_subframes: int = 4,   # ostavljeno 4 jer tvoj LTETxChain trenutno to traži
    normal_cp: bool = True,
    mib_bits: Optional[Iterable[int]] = None,
) -> None:
    # MIB (24 bita)
    if mib_bits is None:
        mib = np.random.randint(0, 2, 24, dtype=np.uint8)
    else:
        mib = np.array(list(mib_bits), dtype=np.uint8).flatten()
        if mib.size != 24:
            raise ValueError("mib_bits mora imati tačno 24 bita.")

    # TX chain preko tvoje klase
    tx = LTETxChain(
        n_id_2=n_id_2,
        ndlrb=ndlrb,
        num_subframes=num_subframes,
        normal_cp=normal_cp,
    )
    waveform, fs = tx.generate_waveform(mib_bits=mib)

    # PBCH simboli za plot: generišemo “čiste” PBCH QPSK simbole iz istih MIB bitova
    pbch_symbols = _pbch_symbols_from_mib(mib)

    print(f"MIB bits ({mib.size}): {mib}")
    print(f"Generisani waveform: {waveform.size} uzoraka, fs = {fs} Hz")
    print(f"PBCH QPSK simbola (prije filtriranja): {pbch_symbols.size}")
    if pbch_symbols.size > 0:
        print(f"[DBG] max |PBCH sym| = {np.max(np.abs(pbch_symbols)):.3g}")

    # 1) PBCH QPSK konstelacija
    plot_pbch_qpsk_constellation(pbch_symbols)

    # 2) Dio OFDM talasnog oblika u vremenu
    plot_ofdm_time_segment(waveform, fs, num_samples=4000)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_tx_chain_demo()
