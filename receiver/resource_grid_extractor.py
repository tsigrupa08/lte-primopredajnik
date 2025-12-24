"""
resource_grid_extractor.py
===========================

Ovaj modul implementira jednostavnu ekstrakciju simbola fizičkog kanala za
emitiranje (PBCH) na strani prijemnika iz LTE resource grid‑a u
frekvencijskom domenu.

PBCH nosi Master Information Block (MIB) i u downlink‑u zauzima centralnih
šest resource blokova (RB) u prvom podokviru. Za normalni ciklički prefiks
(CP) podokvir sadrži četrnaest OFDM simbola, a PBCH obuhvata prva četiri
simbola drugog slota u podokviru 0【868815870562171†L65-L67】. Svaki resource
blok sadrži dvanaest podnosioca, tako da šest centralnih RB‑ova zauzima
ukupno 72 podnosioca. Određeni resource elementi (RE) unutar ovih RB‑ova
rezervisani su za referentne signale specifične za ćeliju (CRS) i druge
kontrolne kanale. Pri generisanju predajničkog grida PBCH simboli se
mapiraju sekvencijalno po frekvenciji za svaki indeks PBCH simbola,
preskačući rezervisane RE. Na prijemniku se obrnuti postupak svodi na
čitanje istih RE iz grida istim redoslijedom.

Ova implementacija pruža jednostavan invertni maper i pretpostavlja:

* **Normalni ciklički prefiks** – prošireni CP (12 simbola po
  podokviru) trenutno nije implementiran.
* **Alokaciju centralnih 6 RB** – PBCH uvijek zauzima šest RB‑ova
  centriranih oko DC podnosioca, bez obzira na ukupnu širinu sistema.

Kada nije zadana eksplicitna maska rezervisanih resource elemenata
(``reserved_re_mask``), ekstraktor čita sve 72 podnosioca za svaki PBCH
OFDM simbol i vraća prvih 240 uzoraka. Ovo odgovara pojednostavljenom
mapiranju korištenom u predajniku u ovom projektu, gdje se PBCH QPSK
simboli postavljaju sekvencijalno bez uračuna referentnih signala. Ako
predajnik preskače određene RE (na primjer zbog CRS‑a ili drugih
kontrolnih kanala), istu booleovu masku treba proslijediti metodi
``extract`` putem argumenta ``reserved_re_mask``. U tom slučaju
ekstraktor će preskočiti RE gdje je maska ``True`` i vratiti sve
preostale PBCH RE.

Klasa ispod vrši ekstrakciju na osnovu dvodimenzionalnog grida. Ne vrši
kanalnu ekvalizaciju ni demodulaciju – jednostavno odabire kompleksne
uzorke koji odgovaraju PBCH RE‑ovima u redoslijedu u kojem su poslani.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class PBCHConfig:
    """Konfiguracija za ekstrakciju PBCH‑a.

    Parametri
    ----------
    ndlrb : int
        Broj downlink resource blokova (NDLRB). Za LTE sistem širine
        1,4 MHz ta vrijednost je 6. PBCH uvijek zauzima centralnih šest
        RB‑ova, neovisno o ukupnoj širini opsega, ali ``ndlrb`` je
        potreban da bi se odredila ukupna širina grida.
    normal_cp : bool
        Postavi na ``True`` za normalni ciklički prefiks (14 OFDM
        simbola po podokviru) ili na ``False`` za prošireni CP (12
        simbola). Trenutno je implementiran samo normalni CP.
    pbch_symbol_indices : Optional[List[int]]
        Lista indeksa OFDM simbola u gridu koji nose PBCH simbole.
        Ako je ``None``, indeksi se automatski određuju na osnovu
        ``normal_cp``; za normalni CP podrazumijevani su ``[7, 8, 9, 10]``.
    pbch_symbols_per_subframe : int
        Broj PBCH QPSK simbola koji se mapiraju u svaki podokvir
        predajnika. U ovom projektu PBCH enkoder generiše 960 QPSK
        simbola koji se šalju kroz četiri podokvira, tako da svaki
        podokvir sadrži 240 simbola. Ova vrijednost se koristi kada
        ekstraktor radi kontiguitetno mapiranje (bez zadate maske
        rezervisanih RE).
    """

    ndlrb: int = 6
    normal_cp: bool = True
    pbch_symbol_indices: Optional[List[int]] = None
    pbch_symbols_per_subframe: int = 240

    def __post_init__(self) -> None:
        # Validate configuration and set defaults
        if not self.normal_cp:
            raise NotImplementedError(
                "Extended cyclic prefix (12 symbols per subframe) is not implemented."
            )
        # Default PBCH symbol indices for normal CP (first four symbols of slot 1)
        if self.pbch_symbol_indices is None:
            # For a normal CP subframe the second slot starts at symbol 7
            object.__setattr__(self, "pbch_symbol_indices", [7, 8, 9, 10])
        else:
            # Copy user‑provided list to avoid accidental external mutation
            object.__setattr__(self, "pbch_symbol_indices", list(self.pbch_symbol_indices))
        # Validate PBCH symbols per subframe
        if self.pbch_symbols_per_subframe <= 0:
            raise ValueError(
                "pbch_symbols_per_subframe must be a positive integer"
            )


class PBCHExtractor:
    """Ekstraktor PBCH simbola iz resource grid‑a u frekvencijskom domenu.

    Ekstraktor čita kompleksne resource elemente koji nose PBCH QPSK
    simbole iz dostavljenog grida. Redoslijed ekstrakcije odgovara
    redoslijedu mapiranja u predajniku: prolazi se kroz OFDM simbole
    određene za PBCH, a unutar svakog simbola prelazi se preko
    frekvencijskih podnosioca preskačući rezervisane elemente.

    Parametri
    ----------
    cfg : PBCHConfig, opcionalno
        Konfiguracija koja određuje lokaciju PBCH‑a i način rezervacije.
        Ako se izostavi, koristi se podrazumijevana konfiguracija za
        ``ndlrb=6`` i normalni CP.

    Primjer
    -------
    >>> import numpy as np
    >>> from rx.resource_grid_extractor import PBCHExtractor, PBCHConfig
    >>> # Kreiraj prazan grid dimenzija (72, 14)
    >>> ndlrb = 6
    >>> grid = np.zeros((12 * ndlrb, 14), dtype=complex)
    >>> # Generiši 240 lažnih PBCH simbola i mapiraj ih sekvencijalno
    >>> pbch = np.arange(240, dtype=complex)
    >>> cfg = PBCHConfig(ndlrb=ndlrb)
    >>> idx = 0
    >>> for l in cfg.pbch_symbol_indices:
    ...     for rb in range(6):
    ...         base = (ndlrb * 12 - 72) // 2 + rb * 12
    ...         for off in range(12):
    ...             if idx >= pbch.size:
    ...                 break
    ...             grid[base + off, l] = pbch[idx]
    ...             idx += 1
    ...         if idx >= pbch.size:
    ...             break
    ...     if idx >= pbch.size:
    ...         break
    >>> # Ekstrakcija PBCH simbola
    >>> extractor = PBCHExtractor(cfg)
    >>> pbch_rx = extractor.extract(grid)
    >>> np.array_equal(pbch_rx, pbch)
    True

    Napomene
    --------
    Ekstraktor ne zavisi od identiteta ćelije niti generisanja CRS‑a; on
    jednostavno čita uzorke iz dostavljenog grida. Ako vaš predajnik
    preskače pojedine resource elemente na osnovu složenije maske
    (npr. CRS, PCFICH ili PHICH), proslijedite tu masku argumentu
    ``reserved_re_mask`` u metodi :func:`extract` kako bi prijemnik
    preskočio iste elemente.
    """

    def __init__(self, cfg: Optional[PBCHConfig] = None) -> None:
        # Use provided configuration or default values
        self.cfg = cfg or PBCHConfig()

    def extract(
        self,
        grid: np.ndarray,
        reserved_re_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Ekstrahuje PBCH simbole iz datog resource grida.

        Parametri
        ----------
        grid : np.ndarray
            Frekvencijski resource grid iz kojeg se PBCH izvlači. Mora
            biti dvodimenzionalni niz oblika ``(num_subcarriers, num_symbols)``,
            gdje je ``num_subcarriers = 12 \times NDLRB``.
        reserved_re_mask : np.ndarray, opcionalno
            Opcionalna booleova maska istih dimenzija kao ``grid``. Mjesta
            gdje je maska ``True`` smatraju se rezervisanim (npr. CRS ili
            drugi kanali) i biće preskočena pri ekstrakciji.

        Povratna vrijednost
        -------------------
        np.ndarray
            Jednodimenzionalni niz kompleksnih vrijednosti koji sadrži
            izdvojene PBCH simbole redanim prema njihovom mapiranju na
            predaji. Za normalni CP ovaj niz sadrži 240 elemenata.

        Izuzeci
        -------
        AssertionError
            Ako ``grid`` nema očekivani broj podnosioca ili nije
            dvodimenzionalan.
        ValueError
            Ako konfiguracija specificira nepodržane parametre.

        Napomene
        --------
        Ne vrši se nikakva demodulacija ni kanalna ekvalizacija. Vraćeni
        simboli su kompleksne vrijednosti iz grida. Pozivatelj je
        odgovoran za daljnju obradu, poput ekvalizacije, demapiranja
        QPSK, deinterleavinga i dekodiranja.
        """
        # Validate grid shape
        if grid.ndim != 2:
            raise AssertionError(
                "grid must be a two‑dimensional array with shape (num_subcarriers, num_symbols)."
            )
        num_subcarriers, num_symbols_total = grid.shape
        expected_subcarriers = 12 * self.cfg.ndlrb
        assert (
            num_subcarriers == expected_subcarriers
        ), f"Grid has {num_subcarriers} subcarriers but expected {expected_subcarriers} (12×NDLRB)."

        # Starting index of the central 6 RBs
        total_sc = num_subcarriers
        pbch_bandwidth = 72  # 6 RB × 12 subcarriers
        k0 = (total_sc - pbch_bandwidth) // 2

        extracted: List[complex] = []

        # Determine extraction pattern
        # If reserved_re_mask is provided, skip REs where mask is True.
        # Otherwise, perform contiguous extraction until pbch_symbols_per_subframe symbols are collected.
        contiguous = reserved_re_mask is None
        symbols_needed = self.cfg.pbch_symbols_per_subframe

        for symbol_index in self.cfg.pbch_symbol_indices:
            if symbol_index < 0 or symbol_index >= num_symbols_total:
                raise ValueError(
                    f"PBCH symbol index {symbol_index} is out of range for a grid with {num_symbols_total} symbols."
                )
            for rb in range(6):
                base = k0 + rb * 12
                for offset in range(12):
                    sc_index = base + offset
                    if reserved_re_mask is not None:
                        # reserved_re_mask is expected to have same shape as grid
                        if reserved_re_mask[sc_index, symbol_index]:
                            continue
                        extracted.append(grid[sc_index, symbol_index])
                        # Do not limit extraction when reserved mask is provided; collect all PBCH REs
                    else:
                        # contiguous mapping: no reserved positions; sequential extraction
                        extracted.append(grid[sc_index, symbol_index])
                        if len(extracted) >= symbols_needed:
                            return np.asarray(extracted, dtype=grid.dtype)

        # If contiguous mapping reached here, return the collected symbols truncated to symbols_needed
        return np.asarray(extracted[:symbols_needed], dtype=grid.dtype)