# receiver/utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class RxValidationConfig:
    """
    Konfiguracija pravila validacije RX signala.

    Attributes
    ----------
    require_complex : bool
        Ako je True, ulaz mora biti kompleksan (complex64/complex128).
    cast_real_to_complex : bool
        Ako je True i require_complex=False, realni signal će se kastovati u kompleksni (imag=0).
    min_num_samples : int
        Minimalan broj uzoraka u vremenskoj osi (zadnja osa).
    eps_power : float
        Minimalna dozvoljena snaga signala (da izbjegnemo dijeljenje sa nulom).
    check_finite : bool
        Ako je True, zabranjuje NaN/Inf vrijednosti.
    out_dtype : np.dtype
        Dtype koji se preferira za izlaz (npr. np.complex64).
    """
    require_complex: bool = True
    cast_real_to_complex: bool = True
    min_num_samples: int = 1
    eps_power: float = 1e-15
    check_finite: bool = True
    out_dtype: np.dtype = np.complex64


class RxUtils:
    """
    RxUtils – pomoćne funkcije za prijemnik: validacija, snaga/RMS,
    normalizacija i konverzije dB↔linear.

    Ova klasa je namjerno "čista" (bez LTE-specifičnih detalja) i služi
    kao stabilna baza za PSS korelaciju, FFT i PBCH ekstrakciju.

    Parameters
    ----------
    config : RxValidationConfig, optional
        Pravila validacije i preferirani dtype.

    Primjeri
    --------
    >>> import numpy as np
    >>> from rx.utils import RxUtils
    >>> u = RxUtils()
    >>> x = (np.random.randn(2048) + 1j*np.random.randn(2048)).astype(np.complex64)
    >>> x = u.validate_rx_samples(x, min_num_samples=128)
    >>> y, scale = u.normalize_rms(x, target_rms=1.0)
    >>> float(u.rms(y))
    1.0
    """

    def __init__(self, config: Optional[RxValidationConfig] = None) -> None:
        self.config = config or RxValidationConfig()

    # ------------------------------------------------------------------
    # Validacija i shape helperi
    # ------------------------------------------------------------------
    def ensure_1d_time_axis(self, x: np.ndarray) -> np.ndarray:
        """
        Osigurava da je signal 1D u vremenu.

        Dozvoljeno:
        - (N,)
        - (1, N)
        - (N, 1)

        Parameters
        ----------
        x : np.ndarray
            Ulazni signal.

        Returns
        -------
        np.ndarray
            1D signal oblika (N,).

        Raises
        ------
        ValueError
            Ako je ulaz matrica (M, N) gdje M>1 i N>1.
        """
        a = np.asarray(x)

        if a.ndim == 1:
            return a

        if a.ndim == 2:
            if a.shape[0] == 1:
                return a.reshape(-1)
            if a.shape[1] == 1:
                return a.reshape(-1)
            raise ValueError("Očekujem vektor (N,), (1,N) ili (N,1), a ne matricu (M,N).")

        raise ValueError("Očekujem 1D ili 2D vektor (N,), (1,N) ili (N,1).")

    def mean_power(self, x: np.ndarray) -> float:
        """
        Računa srednju snagu signala: mean(|x|^2).

        Parameters
        ----------
        x : np.ndarray
            Kompleksni ili realni signal.

        Returns
        -------
        float
            Srednja snaga.
        """
        a = np.asarray(x)
        p = np.mean(np.abs(a) ** 2)
        return float(p)

    def rms(self, x: np.ndarray) -> float:
        """
        RMS signala: sqrt(mean(|x|^2)).

        Parameters
        ----------
        x : np.ndarray
            Signal.

        Returns
        -------
        float
            RMS vrijednost.
        """
        return float(np.sqrt(self.mean_power(x)))

    def validate_rx_samples(
        self,
        rx_samples: np.ndarray,
        require_complex: Optional[bool] = None,
        min_num_samples: Optional[int] = None,
        check_finite: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Validira RX uzorke (dtype/NaN/Inf/min dužina/snaga) i vraća signal
        u kompleksnom dtype-u (po konfiguraciji).

        Napomena: podrazumijevamo da je **zadnja osa vrijeme**.

        Parameters
        ----------
        rx_samples : np.ndarray
            Ulazni signal (1D/2D/ND). Zadnja osa je vrijeme.
        require_complex : bool, optional
            Override za config.require_complex.
        min_num_samples : int, optional
            Override za config.min_num_samples.
        check_finite : bool, optional
            Override za config.check_finite.

        Returns
        -------
        np.ndarray
            Validirani signal, kompleksan dtype (config.out_dtype ili complex128).

        Raises
        ------
        TypeError
            Ako se traži kompleksan signal, a ulaz nije kompleksan.
        ValueError
            Ako signal sadrži NaN/Inf, premalo uzoraka ili praktično nultu snagu.
        """
        cfg = self.config
        req_cplx = cfg.require_complex if require_complex is None else bool(require_complex)
        min_n = cfg.min_num_samples if min_num_samples is None else int(min_num_samples)
        chk_fin = cfg.check_finite if check_finite is None else bool(check_finite)

        x = np.asarray(rx_samples)

        # provjera minimalne dužine (zadnja osa = vrijeme)
        if x.shape[-1] < min_n:
            raise ValueError(f"Premalo uzoraka: {x.shape[-1]} < {min_n}.")

        # dtype provjera + eventualni cast
        is_cplx = np.issubdtype(x.dtype, np.complexfloating)
        is_real = np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer)

        if req_cplx and not is_cplx:
            raise TypeError("rx_samples mora biti kompleksnog tipa (complex64/complex128).")

        if not is_cplx:
            if is_real and cfg.cast_real_to_complex:
                x = x.astype(np.float32, copy=False) + 1j * np.zeros_like(x, dtype=np.float32)
            else:
                raise TypeError("rx_samples mora biti realan ili kompleksan NumPy niz.")

        # konačnost
        if chk_fin:
            if not np.isfinite(x.real).all() or not np.isfinite(x.imag).all():
                raise ValueError("Signal sadrži NaN ili Inf vrijednosti.")

        # snaga
        p = self.mean_power(x)
        if (not np.isfinite(p)) or (p <= cfg.eps_power):
            raise ValueError("Snaga signala je nula ili previše mala; dalja obrada nije stabilna.")

        # dtype izlaza
        out_dtype = cfg.out_dtype
        if out_dtype is not None:
            x = x.astype(out_dtype, copy=False)

        return x

    # ------------------------------------------------------------------
    # Normalizacija i dB helperi
    # ------------------------------------------------------------------
    def normalize_rms(
        self,
        x: np.ndarray,
        target_rms: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Normalizuje signal na ciljni RMS (korisno prije korelacije/PSS detekcije).

        Parameters
        ----------
        x : np.ndarray
            Kompleksni signal.
        target_rms : float
            Ciljni RMS (>0).

        Returns
        -------
        y : np.ndarray
            Normalizovan signal.
        scale : float
            Faktor skale koji je primijenjen (y = x * scale).

        Raises
        ------
        ValueError
            Ako je target_rms <= 0 ili je signal praktično nulte snage.
        """
        if not np.isfinite(target_rms) or target_rms <= 0.0:
            raise ValueError("target_rms mora biti konačan broj > 0.")

        x = np.asarray(x)
        r = self.rms(x)
        if (not np.isfinite(r)) or (r <= self.config.eps_power):
            raise ValueError("RMS je nula ili previše mali za stabilnu normalizaciju.")

        scale = float(target_rms / r)
        y = (x * scale).astype(x.dtype, copy=False)
        return y, scale

    def db2lin(self, db: float) -> float:
        """
        Pretvara dB u linearno (10^(db/10)).

        Parameters
        ----------
        db : float

        Returns
        -------
        float
        """
        return float(10.0 ** (float(db) / 10.0))

    def lin2db(self, x: float, floor_db: Optional[float] = None) -> float:
        """
        Pretvara linearno u dB (10*log10(x)).

        Parameters
        ----------
        x : float
            Linearna vrijednost.
        floor_db : float, optional
            Ako je dato, za x<=0 vraća floor_db umjesto izuzetka.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            Ako x<=0 i floor_db nije zadat.
        """
        x = float(x)
        if x <= 0.0:
            if floor_db is None:
                raise ValueError("lin2db: x mora biti > 0.")
            return float(floor_db)
        return float(10.0 * np.log10(x))



