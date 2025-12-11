import numpy as np


class FrequencyOffset:
    """
    Model frekvencijskog ofseta u baziopojasnom kompleksnom signalu.

    Frekvencijski ofset nastaje zbog razlike između LO (local oscillator)
    frekvencija predajnika i prijemnika. U baziopojasnom domenu modelira se kao
    multiplikacija kompleksnim eksponencijalom:

        r[n] = s[n] * exp( j * 2*pi*Δf * n / f_s )

    gdje je:
        - s[n]  : originalni (TX) kompleksni signal
        - r[n]  : signal sa frekvencijskim ofsetom
        - Δf    : frekvencijski ofset u Hz
        - f_s   : sample rate u Hz
        - n     : indeks uzorka (0, 1, 2, ...)

    Klasa internu pamti indeks uzorka, tako da se više uzastopnih poziva
    `apply` ponašaju kao jedna kontinuirana sekvenca (bez skokova faze).

    Parameters
    ----------
    freq_offset_hz : float
        Frekvencijski ofset Δf u Hz. Pozitivna vrijednost znači da će
        faza signala napredovati u pozitivnom smjeru (rotacija u
        kompleksnoj ravni).
    sample_rate_hz : float
        Sample rate signala f_s u Hz (npr. 1.92e6 za LTE 1.4 MHz ili
        30.72e6 za “punu” LTE konfiguraciju).
    initial_phase_rad : float, optional
        Početna faza (u radijanima) kompleksnog eksponencijala. Default je 0.
    """

    def __init__(
        self,
        freq_offset_hz: float,
        sample_rate_hz: float,
        initial_phase_rad: float = 0.0,
    ) -> None:
        self.freq_offset_hz = float(freq_offset_hz)
        self.sample_rate_hz = float(sample_rate_hz)
        self.initial_phase_rad = float(initial_phase_rad)

        # Interni brojač uzoraka – bitan ako se apply poziva više puta u nizu.
        self._sample_index = 0

    def reset(self) -> None:
        """
        Resetuje interni brojač uzoraka na nulu.

        Koristiti kada želite da simulacija frekvencijskog ofseta krene “iznova”
        od n = 0 (npr. nova nezavisna simulacija).

        Returns
        -------
        None
        """
        self._sample_index = 0

    def _rotation_vector(self, num_samples: int) -> np.ndarray:
        """
        Interna pomoćna funkcija: generiše kompleksni eksponencijal za
        zadani broj uzoraka, uz kontinuitet faze preko više poziva.

        Parameters
        ----------
        num_samples : int
            Broj uzoraka za koje treba izračunati rotaciju.

        Returns
        -------
        np.ndarray
            1D NumPy niz kompleksnih vrijednosti oblika
            ``exp( j * 2*pi*Δf * n / f_s + φ0 )``,
            gdje n počinje od trenutne vrijednosti internog brojača.
        """
        if num_samples <= 0:
            return np.ones(0, dtype=np.complex64)

        # n_global = n0, n0+1, ..., n0+num_samples-1
        n0 = self._sample_index
        n = n0 + np.arange(num_samples, dtype=np.float64)

        # faza: 2*pi*Δf*n / f_s + φ0
        phase = (
            2.0 * np.pi * self.freq_offset_hz * (n / self.sample_rate_hz)
            + self.initial_phase_rad
        )

        rot = np.exp(1j * phase).astype(np.complex64)

        # ažuriraj interni brojač za sljedeći poziv
        self._sample_index += num_samples

        return rot

    def apply(self, tx_samples: np.ndarray) -> np.ndarray:
        """
        Primjenjuje frekvencijski ofset na dati kompleksni TX signal.

        Očekuje se da su uzorci kompleksni (IQ) i da je vremenska osa na
        zadnjoj dimenziji. Funkcija podržava 1D i višedimenzionalne nizove:

        - 1D: shape = (N,)
        - 2D: shape = (n_antena, N)
        - ... općenito: shape = (..., N), gdje je N broj uzoraka.

        Parameters
        ----------
        tx_samples : np.ndarray
            Kompleksni NumPy niz (dtype kompleksan) sa uzorcima signala
            na koje se primjenjuje frekvencijski ofset. Zadnja dimenzija
            se tretira kao vremenski indeks.

        Returns
        -------
        np.ndarray
            Novi NumPy niz iste forme kao `tx_samples`, sa primijenjenim
            frekvencijskim ofsetom.

        Raises
        ------
        ValueError
            Ako `tx_samples` nije barem 1D niz ili nije kompleksnog tipa.

        Examples
        --------
        Jednostavan primjer sa 1D signalom::

            >>> import numpy as np
            >>> from channel.frequency_offset import FrequencyOffset
            >>>
            >>> fs = 1.92e6  # sample rate u Hz
            >>> freq_off = 500.0  # 500 Hz ofset
            >>> fo = FrequencyOffset(freq_offset_hz=freq_off, sample_rate_hz=fs)
            >>>
            >>> # Npr. OFDM izlaz (ovdje samo dummy signal)
            >>> tx = np.ones(1000, dtype=np.complex64)
            >>> rx = fo.apply(tx)
            >>> rx.shape
            (1000,)

        Primjer sa više antena (2D)::

            >>> tx_mimo = np.vstack([tx, 1j * tx])  # shape (2, 1000)
            >>> rx_mimo = fo.apply(tx_mimo)
            >>> rx_mimo.shape
            (2, 1000)
        """
        x = np.asarray(tx_samples)

        if x.ndim < 1:
            raise ValueError("tx_samples mora biti barem 1D NumPy niz.")
        if not np.issubdtype(x.dtype, np.complexfloating):
            raise ValueError("tx_samples mora biti kompleksnog tipa (complex64/complex128).")

        num_samples = x.shape[-1]

        # vektor rotacije dužine N
        rot = self._rotation_vector(num_samples)

        # pripremi shape za broadcasting: (..., N)
        # rot treba da ima shape (1, 1, ..., N)
        rot_shape = (1,) * (x.ndim - 1) + (num_samples,)
        rot = rot.reshape(rot_shape)

        return x * rot
