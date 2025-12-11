import numpy as np


class AWGNChannel:
    """
    Klasa za simulaciju aditivnog bijelog Gaussovog šuma (AWGN)
    na kompleksnom baziopojasnom signalu.

    Šum se modelira kao kompleksni Gaussov šum n[k] sa nultom
    sredinom i varijancom određenom ciljnim SNR-om:

        y[k] = x[k] + n[k]

    gdje je SNR definisan kao:

        SNR = P_signal / P_noise

    u dB:

        SNR_dB = 10 * log10(P_signal / P_noise)

    Za kompleksni signal (I + jQ) varijanca po komponenti je
    P_noise / 2, pa se standardna devijacija šuma računa kao:

        sigma = sqrt(P_noise / 2).

    Parametri
    ---------
    snr_db : float
        Ciljani SNR u decibelima (dB). Definisan kao odnos srednje
        snage signala i snage šuma po kompleksnom uzorku.
    seed : int, optional
        Sjeme za generator slučajnih brojeva. Korisno za
        determinističke (ponovljive) simulacije. Default je None,
        što znači da se koristi globalni RNG state.

    Napomene
    --------
    - Snaga signala procjenjuje se kao srednja vrijednost |x[k]|^2
      preko svih uzoraka.
    - Šum je kompleksan, nezavisan u I i Q komponenti, sa istom
      varijancom u obje komponente.
    """

    def __init__(self, snr_db: float, seed: int | None = None) -> None:
        if not np.isfinite(snr_db):
            raise ValueError("snr_db mora biti konačan broj (nije NaN/inf).")

        self.snr_db = float(snr_db)
        self._rng = np.random.default_rng(seed)

    def _compute_noise_std(self, x: np.ndarray) -> float:
        """
        Interna pomoćna funkcija: računa standardnu devijaciju
        kompleksnog AWGN šuma za zadani signal i ciljni SNR.

        Parametri
        ---------
        x : np.ndarray
            Ulazni signal (kompleksni NumPy niz) čija se snaga koristi
            za izračun šuma.

        Povratna vrijednost
        -------------------
        float
            Standardna devijacija (sigma) za realni i imaginarni dio
            šuma.

        Raises
        ------
        ValueError
            Ako je procijenjena srednja snaga signala praktično nula.
        """
        # globalna srednja snaga signala
        power_signal = np.mean(np.abs(x) ** 2)

        if power_signal <= 0.0 or not np.isfinite(power_signal):
            raise ValueError(
                "Snaga signala je nula ili neispravna; SNR nije definisan."
            )

        snr_linear = 10.0 ** (self.snr_db / 10.0)
        power_noise = power_signal / snr_linear

        # za kompleksni šum: P_noise = 2 * sigma^2
        sigma = np.sqrt(power_noise / 2.0)
        return float(sigma)

    def apply(self, tx_samples: np.ndarray) -> np.ndarray:
        """
        Primjenjuje AWGN na dati kompleksni signal.

        Očekuje se da je `tx_samples` kompleksni NumPy niz bilo kog
        oblika, pri čemu se šum dodaje element-po-element da bi
        izlaz imao isti shape.

        Parametri
        ---------
        tx_samples : np.ndarray
            Kompleksni baziopojasni signal (npr. izlaz OFDM modulatora).
            Može biti 1D (N,), 2D (n_antena, N), ili višedimenzionalan.
            Dtype treba biti kompleksan (complex64 ili complex128).

        Povratna vrijednost
        -------------------
        np.ndarray
            Niz iste dimenzije i dtype-a kao `tx_samples`, sa dodatim
            AWGN šumom.

        Raises
        ------
        ValueError
            Ako `tx_samples` nije kompleksan niz ili ako je snaga
            signala nula (SNR se ne može definisati).

        Primjeri
        --------
        Jednostavan primjer sa 1D signalom::

            >>> import numpy as np
            >>> from channel.awgn_channel import AWGNChannel
            >>>
            >>> # Dummy OFDM talasni oblik
            >>> tx = (np.random.randn(1000) + 1j*np.random.randn(1000)).astype(np.complex64)
            >>>
            >>> ch = AWGNChannel(snr_db=15.0, seed=42)
            >>> rx = ch.apply(tx)
            >>> tx.shape == rx.shape
            True

        Primjer sa više antena (2D)::

            >>> tx_mimo = np.vstack([tx, 1j * tx])   # shape (2, 1000)
            >>> rx_mimo = ch.apply(tx_mimo)
            >>> rx_mimo.shape
            (2, 1000)
        """
        x = np.asarray(tx_samples)

        if not np.issubdtype(x.dtype, np.complexfloating):
            raise ValueError(
                "tx_samples mora biti kompleksnog tipa (complex64/complex128)."
            )

        sigma = self._compute_noise_std(x)

        # generiši šum iste forme kao x
        noise_real = self._rng.standard_normal(size=x.shape, dtype=np.float32)
        noise_imag = self._rng.standard_normal(size=x.shape, dtype=np.float32)
        noise = sigma * (noise_real + 1j * noise_imag)

        # pazi da izlaz zadrži isti dtype
        y = (x + noise).astype(x.dtype, copy=False)
        return y
