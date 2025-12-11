"""
tx_chain_demo
=============

Primjer kompletnog LTE predajnog lanca (TX chain) i vizualizacije:

* QPSK konstelacija PBCH simbola
* dio OFDM talasnog oblika u vremenu (realni dio, imaginarni dio i amplituda)

Skripta koristi klase i funkcije iz paketa ``transmitter``:

* :func:`transmitter.pss.generate_pss_sequence`
* :class:`transmitter.pbch.PBCHEncoder`
* :func:`transmitter.resource_grid.create_resource_grid`
* :func:`transmitter.resource_grid.map_pss_to_grid`
* :func:`transmitter.resource_grid.map_pbch_to_grid`
* :class:`transmitter.ofdm.OFDMModulator`

Kako pokrenuti
--------------

Iz root direktorija projekta (gdje je ``examples/`` na vrhu stabla) pokreni:

.. code-block:: bash

    # Opcija 1 – direktno
    python examples/tx_chain_demo.py

    # Opcija 2 – kao modul (ako je projekat paket)
    python -m examples.tx_chain_demo

Nakon pokretanja će se otvoriti dva prozora sa grafovima
i generisaće se PNG slike u ``examples/``:

1. PBCH QPSK konstelacija (scatter dijagram)
2. Dio OFDM signala u vremenu (realni dio, imaginarni dio i apsolutna vrijednost)

"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------#
# Podešavanje putanje do projekta da bi se "transmitter" paket mogao uvesti
# ---------------------------------------------------------------------------#
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Sada importujemo module iz transmitter/
from transmitter.pss import generate_pss_sequence
from transmitter.resource_grid import (
    create_resource_grid,
    map_pss_to_grid,
    map_pbch_to_grid,
)
from transmitter.pbch import PBCHEncoder
from transmitter.ofdm import OFDMModulator

# Path za snimanje slika direktno u folder examples/
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------#
# Vizualizacija
# ---------------------------------------------------------------------------#
def plot_pbch_qpsk_constellation(pbch_symbols: np.ndarray) -> plt.Figure:
    """
    Crta QPSK konstelaciju PBCH simbola.

    Parametri
    ----------
    pbch_symbols : np.ndarray
        Jednodimenzionalni niz kompleksnih QPSK simbola koji pripadaju
        PBCH-u. Očekuje se da su simboli već normalizirani na
        jediničnu snagu (npr. izlaz :meth:`PBCHEncoder.qpsk_map`).

    Povratna vrijednost
    -------------------
    fig : matplotlib.figure.Figure
        Figure objekt koji sadrži scatter dijagram QPSK konstelacije.

    Napomene
    --------
    Na slici se prikazuju:

    * plave tačke – stvarni PBCH QPSK simboli
    * markeri u obliku 'x' sa oznakama ``00``, ``01``, ``11``, ``10`` –
      idealne QPSK pozicije (Gray mapping)

    Legenda jasno označava šta predstavlja svaka grupa tačaka, a ose
    su označene kao I (in-phase) i Q (quadrature).
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Stvarni PBCH simboli
    ax.scatter(
        pbch_symbols.real,
        pbch_symbols.imag,
        s=16,
        alpha=0.6,
        label="PBCH QPSK simboli",
    )

    # Idealne QPSK tačke (Gray mapping: 00, 01, 11, 10)
    ideal_bits = np.array(
        [
            [0, 0],  # +1 +1
            [0, 1],  # +1 -1
            [1, 1],  # -1 -1
            [1, 0],  # -1 +1
        ],
        dtype=np.uint8,
    )
    encoder = PBCHEncoder(target_bits=16, verbose=False)  # samo da dobijemo map funkciju
    ideal_symbols = encoder.qpsk_map(ideal_bits.flatten())

    labels = ["00", "01", "11", "10"]
    for sym, lab in zip(ideal_symbols, labels):
        ax.scatter(
            sym.real,
            sym.imag,
            marker="x",
            s=80,
            linewidths=2,
            label=f"Idealna tačka {lab}",
        )
        ax.text(
            sym.real * 1.05,
            sym.imag * 1.05,
            lab,
            fontsize=10,
            ha="center",
            va="center",
        )

    ax.set_title("PBCH QPSK konstelacija", fontsize=12)
    ax.set_xlabel("I (in-phase komponenta)")
    ax.set_ylabel("Q (quadrature komponenta)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    # Snimanje slike u examples/
    save_path = os.path.join(EXAMPLES_DIR, "results_pbch_qpsk_constellation.png")
    fig.savefig(save_path, dpi=300)
    print(f"[OK] Spremljena PBCH konstelacija → {save_path}")

    return fig


def plot_ofdm_time_segment(
    waveform: np.ndarray,
    sample_rate: float,
    num_samples: int = 3000,
    start_sample: int = 0,
) -> plt.Figure:
    """
    Crta dio OFDM talasnog oblika u vremenu.

    Parametri
    ----------
    waveform : np.ndarray
        Kompleksni OFDM signal u vremenskoj domeni (izlaz
        :meth:`OFDMModulator.modulate`).
    sample_rate : float
        Frekvencija uzorkovanja signala u Hz.
    num_samples : int, opcionalno
        Koliko uzoraka prikazati na grafiku. Zadano je 3000.
    start_sample : int, opcionalno
        Početni indeks uzorka u signalu od kojeg se uzima segment
        za vizualizaciju. Zadano je 0 (početak signala).

    Povratna vrijednost
    -------------------
    fig : matplotlib.figure.Figure
        Figure objekt sa tri krive:

        * realni dio signala :math:`\\Re\\{s[n]\\}`
        * imaginarni dio signala :math:`\\Im\\{s[n]\\}`
        * amplituda :math:`|s[n]|`

    Napomene
    --------
    Na legendi je eksplicitno naznačeno:

    * "Realni dio s[n]" – puna linija
    * "Imaginarni dio s[n]" – isprekidana linija
    * "Apsolutna vrijednost |s[n]|" – tačkasta linija

    Osa x je u sekundama, tako da se jasno vidi trajanje
    prikazanog segmenta OFDM signala.
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

    # Snimanje slike u examples/
    save_path = os.path.join(EXAMPLES_DIR, "results_ofdm_time_segment.png")
    fig.savefig(save_path, dpi=300)
    print(f"[OK] Spremljen OFDM segment → {save_path}")

    return fig


# ---------------------------------------------------------------------------#
# Minimalni TX chain sklopljen ovdje (bez diranja transmitter/LTETxChain.py)
# ---------------------------------------------------------------------------#
def build_lte_waveform(
    nid2: int = 0,
    ndlrb: int = 6,
    num_subframes: int = 1,
    normal_cp: bool = True,
    mib_bits: Optional[Iterable[int]] = None,
    pbch_target_bits: int = 384,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Generiše LTE OFDM signal i PBCH QPSK simbole koristeći module iz
    paketa ``transmitter`` (bez upotrebe LTETxChain klase).

    Koraci
    ------
    1. Kreiranje praznog LTE resource grida.
    2. Generisanje i mapiranje PSS sekvence.
    3. PBCH enkodiranje MIB bitova i mapiranje u grid.
    4. OFDM modulacija resource grida.

    Parametri
    ----------
    nid2 : int, opcionalno
        LTE fizički identitet ćelije :math:`N_{ID}^{(2)}` (0, 1 ili 2).
    ndlrb : int, opcionalno
        Broj downlink resource blokova :math:`N_{DL}^{RB}`. Za 1.4 MHz je 6.
    num_subframes : int, opcionalno
        Broj subfrejmova u resource gridu. Zadano 1.
    normal_cp : bool, opcionalno
        Ako je True, koristi se normal CP (14 simbola po subfrejmu),
        u suprotnom extended CP (12 simbola).
    mib_bits : Iterable[int], opcionalno
        Binarni niz MIB informacija za PBCH. Ako je ``None``, generiše se
        nasumičnih 24 bita.
    pbch_target_bits : int, opcionalno
        Broj bitova nakon rate matchinga u PBCH enkoderu (E). Zadano je 384.

    Povratna vrijednost
    -------------------
    waveform : np.ndarray
        Kompleksni OFDM signal u vremenskoj domeni (sa CP).
    fs : float
        Sample rate signala u Hz.
    pbch_symbols : np.ndarray
        QPSK PBCH simboli (kompleksni).
    mib_bits_out : np.ndarray
        Korišteni MIB bitovi (24 bita) u obliku vektora tipa uint8.
    """
    # 1) Resource grid
    grid = create_resource_grid(
        ndlrb=ndlrb,
        num_subframes=num_subframes,
        normal_cp=normal_cp,
    )

    # 2) PSS generacija i mapiranje (PSS dužine 62)
    pss = generate_pss_sequence(nid2)
    map_pss_to_grid(grid, pss, symbol_index=6, ndlrb=ndlrb)

    # 3) PBCH enkodiranje MIB bitova
    if mib_bits is None:
        mib_bits = np.random.randint(0, 2, 24, dtype=np.uint8)
    else:
        mib_bits = np.array(list(mib_bits), dtype=np.uint8).flatten()

    pbch_encoder = PBCHEncoder(target_bits=pbch_target_bits, verbose=False)
    pbch_symbols = pbch_encoder.encode(mib_bits)

    # Mapiranje PBCH u grid – edukativno: simboli 0,1,2,3 prvog subfrejma
    map_pbch_to_grid(
        grid,
        pbch_symbols,
        pbch_symbol_indices=[0, 1, 2, 3],
        ndlrb=ndlrb,
    )

    # 4) OFDM modulacija
    modulator = OFDMModulator(grid)
    waveform, fs = modulator.modulate()

    return waveform, fs, pbch_symbols, mib_bits


# ---------------------------------------------------------------------------#
# Glavni demo
# ---------------------------------------------------------------------------#
def run_tx_chain_demo(
    nid2: int = 0,
    ndlrb: int = 6,
    num_subframes: int = 1,
    normal_cp: bool = True,
) -> None:
    """
    Pokreće kompletan LTE TX chain (napravljen u ovom modulu) i
    prikazuje vizualizacije PBCH konstelacije i OFDM talasnog oblika.

    Parametri
    ----------
    nid2 : int, opcionalno
        LTE fizički identitet ćelije :math:`N_{ID}^{(2)}` (0, 1 ili 2).
    ndlrb : int, opcionalno
        Broj downlink resource blokova :math:`N_{DL}^{RB}`.
    num_subframes : int, opcionalno
        Broj LTE subfrejmova u resource gridu.
    normal_cp : bool, opcionalno
        Označava da li se koristi normal CP (True) ili extended CP (False).

    Povratna vrijednost
    -------------------
    None
        Funkcija nema povratnu vrijednost; prikazuje dva grafička prozora.

    Primjer
    -------
    Jednostavno pokretanje sa podrazumijevanim parametrima::

        >>> run_tx_chain_demo()

    ili iz komandne linije::

        python examples/tx_chain_demo.py
    """
    waveform, fs, pbch_symbols, mib_bits = build_lte_waveform(
        nid2=nid2,
        ndlrb=ndlrb,
        num_subframes=num_subframes,
        normal_cp=normal_cp,
    )

    print(f"MIB bits ({len(mib_bits)}): {mib_bits}")
    print(f"Generisani waveform dužine {waveform.size} uzoraka pri fs = {fs} Hz")
    print(f"PBCH QPSK simbola: {pbch_symbols.size}")

    # 1) PBCH QPSK konstelacija
    plot_pbch_qpsk_constellation(pbch_symbols)

    # 2) Dio OFDM talasnog oblika u vremenu
    plot_ofdm_time_segment(waveform, fs, num_samples=4000)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_tx_chain_demo()
