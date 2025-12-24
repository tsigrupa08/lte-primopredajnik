"""
pss_demo.py

Primjer generisanja i vizualizacije LTE Primary Synchronization Signal (PSS)
sekvenci za N_ID_2 = 0, 1 i 2. Prikazuju se magnitude i faze Zadoff–Chu
sekvenci definisanih u 3GPP TS 36.211, sekcija 6.11.

Skripta služi kao demonstracija pravilnog generisanja PSS signala
korištenjem funkcije `generate_pss_sequence()` iz modula
`transmitter.pss`.

Notes
-----
PSS je Zadoff–Chu sekvenca dužine 62. Postoje tačno tri moguće PSS
sekvence koje zavise od vrijednosti N_ID_2 ∈ {0, 1, 2}.
Svaka sekvenca ima različit korijen ZC sekvence (u = 25, 29 ili 34).

Run
---
Pokrenuti iz root foldera projekta:

    python examples/pss_demo.py

Examples
--------
Generisanje jedne PSS sekvence (npr. N_ID_2 = 1):

>>> from transmitter.pss import generate_pss_sequence
>>> pss = generate_pss_sequence(1)
>>> len(pss)
62
>>> abs(pss[0])
1.0

Pokretanje vizualizacije iz komandne linije:

    python examples/pss_demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from transmitter.pss import generate_pss_sequence


def main() -> None:
    """
    Generiše i vizualizira sve tri LTE PSS sekvence (N_ID_2 = 0, 1, 2).

    Funkcija generiše PSS sekvence, provjerava dužinu rezultata, te
    prikazuje dvije vrste grafova:
    1. Magnituda |d[n]| za sve tri Zadoff–Chu sekvence.
    2. Faza ϕ[n] = angle(d[n]) u radijanima.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Funkcija nema povratnu vrijednost. Kao rezultat, generiše dve
        matplotlib figure prikazane na ekranu.

    Raises
    ------
    ValueError
        Ako generisana PSS sekvenca nema dužinu tačno 62 uzorka.

    Notes
    -----
    - Sekvence su kompleksne i jedinične magnitude.
    - MATLAB ekvivalent definisan je u referentnoj funkciji
      `GeneratePSS_Sequence.m` iz pratećeg materijala.

    """

    # Tri moguće PSS sekvence (N_ID_2 = 0, 1, 2)
    nid2_values = [0, 1, 2]

    pss_sequences = []
    for nid2 in nid2_values:
        seq = generate_pss_sequence(nid2)
        if seq.shape[0] != 62:
            raise ValueError(
                f"Očekivana dužina PSS sekvence je 62, a dobijeno {seq.shape[0]}"
            )
        pss_sequences.append(seq)

    # Indeksi uzoraka d[n]
    n = np.arange(62)

    # ----------------------------------------------
    # 1) Magnituda |d[n]|
    # ----------------------------------------------
    fig_mag, axes_mag = plt.subplots(
        len(nid2_values), 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    if len(nid2_values) == 1:
        axes_mag = [axes_mag]

    for ax, seq, nid2 in zip(axes_mag, pss_sequences, nid2_values):
        magnitude = np.abs(seq)
        ax.stem(n, magnitude)
        ax.set_ylabel("|d[n]|")
        ax.set_title(f"PSS magnituda (N_ID_2 = {nid2})")
        ax.grid(True, which="both", linestyle=":")

    axes_mag[-1].set_xlabel("n (indeks uzorka)")
    fig_mag.suptitle("Magnitude LTE PSS sekvenci", fontsize=14)

    # ----------------------------------------------
    # 2) Faza ∠ d[n]
    # ----------------------------------------------
    fig_phase, axes_phase = plt.subplots(
        len(nid2_values), 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )

    if len(nid2_values) == 1:
        axes_phase = [axes_phase]

    for ax, seq, nid2 in zip(axes_phase, pss_sequences, nid2_values):
        phase = np.angle(seq)
        ax.stem(n, phase)
        ax.set_ylabel("ϕ[n] (rad)")
        ax.set_title(f"PSS faza (N_ID_2 = {nid2})")
        ax.grid(True, which="both", linestyle=":")

    axes_phase[-1].set_xlabel("n (indeks uzorka)")
    fig_phase.suptitle("Faze LTE PSS sekvenci", fontsize=14)

    # ------------------------------------------------------
    # Snimanje slika u projektni folder examples/figures
    # ------------------------------------------------------
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)

    output_mag = os.path.join(figures_dir, "pss_demo_magnitude.png")
    output_phase = os.path.join(figures_dir, "pss_demo_phase.png")

    fig_mag.savefig(output_mag, dpi=300)
    fig_phase.savefig(output_phase, dpi=300)

    print(f"[INFO] Sačuvano: {output_mag}")
    print(f"[INFO] Sačuvano: {output_phase}")

    plt.show()


if __name__ == "__main__":
    main()
