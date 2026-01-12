"""
resource_grid_extractor.py
===========================

Ekstrakcija PBCH simbola iz LTE resource grid-a (RX strana).

Ovaj extractor je usklađen sa tvojim TX mapiranjem:
- PBCH se mapira sekvencijalno po frekvenciji unutar centralnih 6 RB (72 subcarrier-a)
  za svaki PBCH OFDM simbol, preskačući reserved RE ako je maska zadana.
- TX staje nakon što potroši tačno onoliko simbola koliko je dobio (npr. 240 po subfrejmu).

Zato i RX extractor MORA:
- preskakati reserved RE na isti način,
- ali uvijek vratiti TAČNO pbch_symbols_to_extract simbola (npr. 240 ili 960),
  tj. stati čim ih skupi (bez obzira ima li maske).

Napomena:
- Grid očekujemo u obliku (subcarriers, symbols) kao u tvom TX resource_grid.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def pbch_symbol_indices_for_subframes(
    num_subframes: int,
    normal_cp: bool = True,
    start_subframe: int = 0,
) -> List[int]:
    """
    Helper: napravi globalne OFDM symbol indekse za PBCH kroz više subfrejmova.

    Normal CP: PBCH u subfrejmu s počinje na l = base + [7,8,9,10]
    Extended CP: PBCH u subfrejmu s počinje na l = base + [6,7,8,9]

    base = (start_subframe + s) * symbols_per_subframe
    symbols_per_subframe = 14 (normal) ili 12 (extended)
    """
    if num_subframes <= 0:
        raise ValueError("num_subframes mora biti > 0")

    symbols_per_subframe = 14 if normal_cp else 12
    if normal_cp:
        local = [7, 8, 9, 10]
    else:
        local = [6, 7, 8, 9]

    idx: List[int] = []
    for sf in range(start_subframe, start_subframe + num_subframes):
        base = sf * symbols_per_subframe
        idx.extend([base + l for l in local])
    return idx


@dataclass(frozen=True)
class PBCHConfig:
    """
    Konfiguracija ekstrakcije.

    ndlrb:
        Ukupan broj DL RB u gridu (za 1.4 MHz je 6).
        Koristi se samo za sanity-check; realno se sve može zaključiti iz grid.shape.
    normal_cp:
        True -> 14 simbola/subfrejm, False -> 12 simbola/subfrejm.
    pbch_symbol_indices:
        Globalni indeksi OFDM simbola (kolone) iz kojih se vadi PBCH.
        Ako je None:
            normal_cp -> [7,8,9,10]
            extended  -> [6,7,8,9]
    pbch_symbols_to_extract:
        TAČAN broj PBCH QPSK simbola koji želiš izvući.
        - za 1 subfrejm (tvoj chunk): 240
        - za 4 subfrejma (tvoj puni PBCH): 960
        - za N subfrejmova: 240*N
    """
    ndlrb: int = 6
    normal_cp: bool = True
    pbch_symbol_indices: Optional[List[int]] = None
    pbch_symbols_to_extract: int = 240

    def __post_init__(self) -> None:
        if self.pbch_symbol_indices is None:
            if self.normal_cp:
                object.__setattr__(self, "pbch_symbol_indices", [7, 8, 9, 10])
            else:
                object.__setattr__(self, "pbch_symbol_indices", [6, 7, 8, 9])
        else:
            object.__setattr__(self, "pbch_symbol_indices", list(self.pbch_symbol_indices))

        if self.pbch_symbols_to_extract <= 0:
            raise ValueError("pbch_symbols_to_extract mora biti pozitivan cijeli broj.")


class PBCHExtractor:
    def __init__(self, cfg: Optional[PBCHConfig] = None) -> None:
        self.cfg = cfg or PBCHConfig()

    def extract(
        self,
        grid: np.ndarray,
        reserved_re_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Ekstrahuje PBCH simbole iz resursnog grida.

        grid: shape (num_subcarriers, num_symbols_total)
        reserved_re_mask: optional bool maska istog shape-a kao grid.
                          True znači "rezervisano" (preskoči).
        """
        grid = np.asarray(grid)
        if grid.ndim != 2:
            raise ValueError("grid mora biti 2D matrica: (subcarriers, symbols).")

        num_subcarriers, num_symbols_total = grid.shape

        # Sanity-check NDLRB ako user želi (ne mora, ali pomaže)
        if self.cfg.ndlrb is not None:
            expected = 12 * int(self.cfg.ndlrb)
            if expected != num_subcarriers:
                # Ne rušimo nužno, ali je vrlo vjerovatno greška u spajanju pipeline-a
                raise ValueError(
                    f"grid ima {num_subcarriers} subcarrier-a, a cfg.ndlrb={self.cfg.ndlrb} očekuje {expected}."
                )

        if reserved_re_mask is not None:
            reserved_re_mask = np.asarray(reserved_re_mask).astype(bool)
            if reserved_re_mask.shape != grid.shape:
                raise ValueError("reserved_re_mask mora imati isti shape kao grid.")

        # ---------------- Centralnih 6 RB (72 subcarrier-a) ----------------
        pbch_bandwidth = 72  # 6 RB × 12
        if num_subcarriers < pbch_bandwidth:
            raise ValueError("grid nema dovoljno subcarrier-a za centralnih 6 RB (72).")

        k0 = (num_subcarriers - pbch_bandwidth) // 2  # start index centralnog 72-opsega

        symbols_needed = int(self.cfg.pbch_symbols_to_extract)
        extracted: List[complex] = []

        # ---------------- Ekstrakcija (isti redoslijed kao TX mapiranje) ----------------
        for symbol_index in self.cfg.pbch_symbol_indices:
            if symbol_index < 0 or symbol_index >= num_symbols_total:
                raise ValueError(
                    f"PBCH simbol indeks {symbol_index} je van opsega 0..{num_symbols_total - 1}."
                )

            for rb in range(6):
                base = k0 + rb * 12
                for offset in range(12):
                    sc_index = base + offset

                    if reserved_re_mask is not None and reserved_re_mask[sc_index, symbol_index]:
                        continue

                    extracted.append(grid[sc_index, symbol_index])

                    # KLJUČNA POPRAVKA: uvijek stani kad skupiš traženi broj simbola
                    if len(extracted) >= symbols_needed:
                        return np.asarray(extracted, dtype=grid.dtype)

        # Ako nismo uspjeli skupiti dovoljno simbola, to znači da:
        # - pbch_symbol_indices nisu pokrili dovoljno RE,
        # - ili reserved_re_mask preskače previše,
        # - ili grid nije popunjen kako očekujemo.
        raise ValueError(
            f"Nije moguće izvući traženih {symbols_needed} PBCH simbola. "
            f"Izvučeno je {len(extracted)}. Provjeri pbch_symbol_indices / masku / num_subframes."
        )


# -------------------------- Brzi primjer upotrebe --------------------------
if __name__ == "__main__":
    # primjer: 1.4 MHz, normal CP, 4 subfrejma => 960 simbola
    # grid = ... (72, 56) ako imaš 4 subfrejma (4*14 simbola)
    pass
